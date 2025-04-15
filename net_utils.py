from utils import evaluate_segmentation
import os
import copy
import time
import uuid
import torch

def train_net(mode, root, model, criterion, optimizer, dataloaders, num_epochs=100, patience=10, device='cpu', run=None, inferer=None, num_classes=1):
    train_dataloader = dataloaders[0]
    val_dataloader = dataloaders[1]
    
    best_val_loss = float('inf')
    best_val_metrics = {"IoU": 0, "Dice": 0}

    best_model_path = os.path.join(root, "models", f"best_model_{uuid.uuid4()}.pth")
    early_stopping_counter = 0


    for epoch in range(num_epochs):
        start_time = time.time()
    
        model.train()
        train_dataloader.dataset.dataset.set_mode(train_mode=True)
        total_loss = 0
        total_iou = 0
        total_dice = 0
        total_tpr = 0
        total_precision = 0
        num_batches = len(train_dataloader)

        for img_patch, mask_patch, image, mask, _ in train_dataloader:
            if mode == '2d':
                inputs = [img.unsqueeze(1).to(device, dtype=torch.float32) for img in image]   # (D, 1, H, W)
                targets = [msk.unsqueeze(1).to(device, dtype=torch.long) for msk in mask]   # (D, 1, H, W)
                inputs = torch.cat(inputs, dim=0)  # (sum D, 1, H, W)
                targets = torch.cat(targets, dim=0)
            elif mode == '3d':
                inputs, targets = img_patch.to(device, dtype=torch.float32), mask_patch.to(device, dtype=torch.long)
                if inputs.shape[1] == 1 and len(inputs.shape) == 5:
                    inputs = inputs.permute(1, 0, 2, 3, 4)
                    targets = targets.permute(1, 0, 2, 3, 4)
                elif inputs.shape[1] != 1 and len(inputs.shape) == 5:
                    inputs = inputs.reshape(inputs.shape[0] * inputs.shape[1], *inputs.shape[2:]).unsqueeze(1)
                    targets = targets.reshape(targets.shape[0] * targets.shape[1], *targets.shape[2:]).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(inputs)
            criterion = criterion.to(device=logits.device)
            loss = criterion(logits, targets)
            metrics = evaluate_segmentation(logits, targets, num_classes=num_classes)
            total_loss += loss.detach().item()
            total_iou += metrics["IoU"]
            total_dice += metrics["Dice"]
            total_tpr += metrics["TPR"]
            total_precision += metrics["Precision"]
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        avg_dice = total_dice / num_batches
        avg_tpr = total_tpr / num_batches
        avg_precision = total_precision / num_batches

        if run:
            run["train/loss"].log(avg_loss)
            run["train/IoU"].log(avg_iou)
            run["train/Dice"].log(avg_dice)
            run["train/TPR"].log(avg_tpr)
            run["train/Precision"].log(avg_precision)

        model.eval()
        val_dataloader.dataset.dataset.set_mode(train_mode=False)
        val_loss = 0
        val_iou = 0
        val_dice = 0
        val_tpr = 0
        val_precision = 0
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
                    targets = mask.to(device, dtype=torch.long)
                    targets = targets.to(torch.device('cpu'))
                    inputs = inputs.unsqueeze(0)
                    targets = targets.unsqueeze(0)
                    logits = inferer(inputs=inputs, network=model)

                metrics = evaluate_segmentation(logits, targets, num_classes=num_classes, prob_thresh=0.5)
                criterion = criterion.to(device=logits.device)
                loss = criterion(logits,  targets)
                val_loss += loss.detach().item()
                val_iou += metrics["IoU"]
                val_dice += metrics["Dice"]
                val_tpr += metrics["TPR"]
                val_precision += metrics["Precision"]

                current_patient_metrics[str(id[0])] = {
                    "Loss": loss.item(),
                    "IoU": metrics["IoU"],
                    "Dice": metrics["Dice"],
                    "TPR": metrics["TPR"],
                    "Precision": metrics["Precision"]
                }

        avg_val_loss = val_loss / num_val_batches
        avg_val_iou = val_iou / num_val_batches
        avg_val_dice = val_dice / num_val_batches
        avg_val_tpr = val_tpr / num_val_batches
        avg_val_precision = val_precision / num_val_batches

        if run:
            run["val/loss"].log(avg_val_loss)
            run["val/IoU"].log(avg_val_iou)
            run["val/Dice"].log(avg_val_dice)
            run["val/TPR"].log(avg_val_tpr)
            run["val/Precision"].log(avg_val_precision)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_metrics = {"IoU": avg_val_iou, "Dice": avg_val_dice}
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
        print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_loss:.4f}, Train IoU: {avg_iou:.4f}, Train Dice: {avg_dice:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val Dice: {avg_val_dice:.4f}, "
          f"Time: {epoch_time_hms}")

    print(f"Saved best model with Val Loss: {best_val_loss:.4f}, Val IoU: {best_val_metrics['IoU']:.4f}, Val Dice: {best_val_metrics['Dice']:.4f}")
    torch.save(best_model, best_model_path)
    
    print("\nBest Patient-wise Metrics (when Val Loss was lowest):")
    for patient_id, metrics in best_patient_metrics.items():
        print(
            f"Patient_ID: {patient_id:<10} | "
            f"Loss: {metrics['Loss']:.4f} | "
            f"IoU: {metrics['IoU']:.3f} | "
            f"Dice: {metrics['Dice']:.3f} | "
            f"TPR: {metrics['TPR']:.3f} | "
            f"Precision: {metrics['Precision']:.3f}"
        )


    if run:
        run["model_filename"] = best_model_path

    return best_model_path




def test_net(mode, model, best_model_path, test_dataloader, device, num_classes=1, run=None, inferer=None):

    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)
    model.eval()

    total_iou = 0
    total_dice = 0
    num_samples = len(test_dataloader)
    probs = 0.5

    with torch.no_grad():
        test_dataloader.dataset.dataset.set_mode(train_mode=False)
        for i, (_, _, image, mask, _) in enumerate(test_dataloader):
            inputs = image.to(device, dtype=torch.float32)
            targets = mask.to(device, dtype=torch.long)

            if mode ==  '2d':
                inputs = inputs.permute(1, 0, 2, 3)
                targets = targets.permute(1, 0, 2, 3)
                logits = model(inputs)
            elif mode =='3d':
                inputs = inputs.unsqueeze(0)
                targets = targets.unsqueeze(0)
                targets = targets.to(torch.device('cpu'))
                logits = inferer(inputs=inputs, network=model)
            
            metrics = evaluate_segmentation(logits, targets, num_classes=num_classes, prob_thresh=probs)
            
            total_iou += metrics["IoU"]
            total_dice += metrics["Dice"]

    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples

    if run:
        run["test/avg_IoU"] = avg_iou
        run["test/avg_Dice"] = avg_dice

    print(f"Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}")