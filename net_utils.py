from utils import evaluate_segmentation
import os
import copy
import time
import uuid
import torch

def train_net(mode, root, model, criterion, optimizer, scheduler, dataloaders, num_epochs=100, patience=10, device='cpu', run=None, inferer=None, num_classes=1):
    train_dataloader = dataloaders[0]
    val_dataloader = dataloaders[1]
    
    best_val_loss = float('inf')
    best_val_metrics = {"IoU": 0., "Dice": 0.}

    best_model_path = os.path.join(root, "models", f"best_model_{uuid.uuid4()}.pth")
    early_stopping_counter = 0


    for epoch in range(num_epochs):
        start_time = time.time()
    
        model.train()
        train_dataloader.dataset.dataset.set_mode(train_mode=True)
        total_loss = 0
        totals = None
        num_batches = len(train_dataloader)

        for img_patch, mask_patch, image, mask, _ in train_dataloader:
            if mode == '2d':
                inputs = [img.unsqueeze(1).to(device, dtype=torch.float32) for img in image]   # (D, 1, H, W)
                targets = [msk.unsqueeze(1).to(device, dtype=torch.long) for msk in mask]   # (D, 1, H, W)
                inputs = torch.cat(inputs, dim=0)  # (sum D, 1, H, W)
                targets = torch.cat(targets, dim=0)
            elif mode == '3d':
                inputs, targets = img_patch.to(device, dtype=torch.float32), mask_patch.to(device, dtype=torch.long)
                inputs = inputs.reshape(inputs.shape[0] * inputs.shape[1], *inputs.shape[2:]).unsqueeze(1)
                targets = targets.reshape(targets.shape[0] * targets.shape[1], *targets.shape[2:]).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(inputs)
            criterion = criterion.to(device=logits.device)
            targets = targets.float()
            loss = criterion(logits, targets)
            metrics = evaluate_segmentation(logits, targets, epoch)
            total_loss += loss.detach().item()
            if totals is None:
                totals = {key: 0. for key in metrics.keys()}
            for key, value in metrics.items():
                totals[key] += value
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / num_batches
        averages = {key: total / num_batches for key, total in totals.items() if 'patient' not in key}

        if run:
            run["train/loss"].log(avg_loss)
            for key, avg in averages.items():
                run[f"train/avg_{key}"].log(avg)

        model.eval()
        val_dataloader.dataset.dataset.set_mode(train_mode=False)
        val_loss = 0
        val_totals = None
        num_val_batches = len(val_dataloader)
        
        current_patient_metrics = {}
        with torch.no_grad():
            for img_patch, mask_patch, image, mask, id in val_dataloader:
                if mode == '2d':
                    inputs = image.to(device, dtype=torch.float32)
                    targets = mask.to(device, dtype=torch.long)
                    inputs = inputs.permute(1, 0, 2, 3)
                    targets = targets.permute(1, 0, 2, 3)
                    logits = model(inputs)
                elif mode == '3d':
                    inputs = image.to(device, dtype=torch.float32)
                    targets = mask["mask"].to(torch.device('cpu'), dtype=torch.long)
                    body_mask = mask["body_mask"].to(torch.device('cpu'), dtype=torch.long)
                    inputs = inputs.unsqueeze(0)
                    targets = targets.unsqueeze(0)
                    body_mask = body_mask.unsqueeze(0)
                    logits = inferer(inputs=inputs, network=model)
                    logits[body_mask == 0] = -1e10
                metrics = evaluate_segmentation(logits, targets, epoch)
                criterion = criterion.to(device=logits.device)
                targets = targets.float()
                loss = criterion(logits, targets)
                val_loss += loss.detach().item()
                if val_totals is None:
                    val_totals = {key: 0. for key in metrics.keys()}
                for key, value in metrics.items():
                    val_totals[key] += value
                current_patient_metrics[str(id[0])] = {
                    "Loss": loss.item(),
                    "IoU": metrics["IoU"],
                    "Dice": metrics["Dice"],
                    "TPR": metrics["TPR"],
                    "Precision": metrics["Precision"],
                }
        

        avg_val_loss = val_loss / num_val_batches
        val_averages = {key: total / num_val_batches for key, total in val_totals.items() if 'patient' not in key}

        if scheduler is not None:
            scheduler.step(avg_loss)

        if run:
            run["val/loss"].log(avg_val_loss)
            for key, avg in val_averages.items():
                run[f"val/avg_{key}"].log(avg)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_metrics = {metric: val_averages[metric] for metric in ["IoU", "Dice"]}
            best_patient_metrics = current_patient_metrics.copy()

            if run:
                run["val/best_val_metrics/IoU"] = best_val_metrics['IoU']
                run["val/best_val_metrics/Dice"] = best_val_metrics['Dice']

            best_model = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if run:
            run["val/patience_counter"] = early_stopping_counter

        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        end_time = time.time()
        epoch_time = end_time - start_time

        epoch_time_hms = time.strftime("%H:%M:%S", time.gmtime(epoch_time))
        metrics_str = f"Train Loss: {avg_loss:.4f} | Train IoU: {averages['IoU']:.4f} | Train Dice: {averages['Dice']:.4f} | " \
                  f"Val Loss: {avg_val_loss:.4f} | Val IoU: {val_averages['IoU']:.4f} | Val Dice: {val_averages['Dice']:.4f} | " \
                  f"Epoch Time: {epoch_time_hms}"
        print(f"Epoch [{epoch+1}/{num_epochs}] | {metrics_str}")

    print(f"Saved best model with Val Loss: {best_val_loss:.4f}, Val IoU: {best_val_metrics['IoU']:.4f}, "
        f"Val Dice: {best_val_metrics['Dice']:.4f}")
    torch.save(best_model, best_model_path)
    
    print("\nBest Patient-wise Metrics (when Val Loss was lowest):")
    for patient_id, metric in best_patient_metrics.items():
        print(
            f"Patient_ID: {patient_id:<10} | "
            f"Loss: {metric['Loss']:.4f} | "
            f"IoU: {metric['IoU']:.3f} | "
            f"Dice: {metric['Dice']:.3f} | "
            f"TPR: {metric['TPR']:.3f} | "
            f"Precision: {metric['Precision']:.3f} | "
        )

    if run:
        run["model_filename"] = best_model_path

    return best_model_path




def test_net(mode, model, best_model_path, test_dataloader, device, num_classes=1, run=None, inferer=None):

    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)
    model.eval()

    totals = None
    num_samples = len(test_dataloader)
    probs = 0.5

    with torch.no_grad():
        test_dataloader.dataset.dataset.set_mode(train_mode=False)
        for i, (_, _, image, mask, _) in enumerate(test_dataloader):
            if mode == '2d':
                inputs = image.to(device, dtype=torch.float32)
                targets = mask.to(device, dtype=torch.long)
                inputs = inputs.permute(1, 0, 2, 3)
                targets = targets.permute(1, 0, 2, 3)
                logits = model(inputs)
            elif mode == '3d':
                inputs = image.to(device, dtype=torch.float32)
                targets = mask['mask'].to(device, dtype=torch.long)
                body_mask = mask["body_mask"].to(torch.device('cpu'), dtype=torch.long)
                inputs = inputs.unsqueeze(0)
                targets = targets.unsqueeze(0)
                body_mask = body_mask.unsqueeze(0)
                targets = targets.to(torch.device('cpu'))
                logits = inferer(inputs=inputs, network=model)
                logits[body_mask == 0] = -1e10

            metrics = evaluate_segmentation(logits, targets, num_classes=num_classes, prob_thresh=probs)

            if totals is None:
                totals = {key: 0. for key in metrics.keys()}
            for key, value in metrics.items():
                totals[key] += value

    averages = {key: total / num_samples for key, total in totals.items() if 'patient' not in key}
    avg_metrics_str = ", ".join([f"Average {key}: {avg:.4f}" for key, avg in averages.items()])

    if run:
        for key, avg in averages.items():
            run[f"test/avg_{key}"] = avg

    print(avg_metrics_str)
