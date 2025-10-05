import os
import time
import torch
import statistics
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

from utils.tools import *
from loss.WBCE_Dice import Dice_CE_Loss
from utils.scheduler import WarmupScheduler
from utils.metrics import Evaluator
from model.net import Net


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_fn(args, model, data_loader, loss_fn, optimizer):
    model.train()
    loop = tqdm(data_loader)
    avg_loss = 0.0
    loss_list = []
    size = (args.img_height, args.img_width)
    interp = nn.Upsample(size=size, mode='bilinear', align_corners=False)

    for index, (image, mask) in enumerate(loop):
        optimizer.zero_grad()
        image = image.to(DEVICE)
        mask = mask.long().to(DEVICE)

        pred = model(image)
        pred = interp(pred)
        loss = loss_fn(pred, mask)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

        loop.update(1)
        avg_loss = statistics.mean(loss_list)
        loop.set_description(f"loss = {avg_loss:.4f}")

    loop.close()
    time.sleep(0.1)

    return avg_loss


def check_metric(args, loader, model):
    model.eval()
    loop = tqdm(loader)
    loop.set_description("Val process")
    evaluator = Evaluator(2)
    metric_dict = {}
    size = (args.img_height, args.img_width)
    interp = nn.Upsample(size=size, mode='bilinear', align_corners=False)

    with torch.no_grad():
        for img, mask in loop:
            img = img.to(DEVICE)
            outputs = model(img)
            outputs = interp(outputs)
            pred = outputs.data.max(1)[1].cpu().numpy()
            evaluator.add_batch(mask.cpu().numpy(), pred)

    metric_dict['acc'] = evaluator.pixel_accuracy()
    metric_dict['acc_class'] = evaluator.pixel_accuracy_class()
    metric_dict['mIoU'] = evaluator.mean_intersection_over_union()
    metric_dict['fwavacc'] = evaluator.frequency_weighted_intersection_over_union()
    metric_dict['iou'] = evaluator.intersection_over_union_crack()
    print(f"acc: {metric_dict['acc']}, iou: {metric_dict['iou']}, acc_class: {metric_dict['acc_class']}, " 
          f"mIoU: {metric_dict['mIoU']}, fwavacc: {metric_dict['fwavacc']}")

    return metric_dict


def main(args):
    train_transform, val_transform = get_transform(args)
    train_loader = get_loader(img_dir=args.source_train_img_dir, mask_dir=args.source_train_mask_dir,
                              batch_size=args.batch_size, num_workers=args.num_workers, transform=train_transform)
    val_loader = get_loader(img_dir=args.source_val_img_dir, mask_dir=args.source_val_mask_dir,
                            batch_size=args.batch_size_val, num_workers=args.num_workers, transform=val_transform)
    model = Net(num_classes=2, init_weights_path=args.init_weights_path).to(DEVICE)
    loss_fn = Dice_CE_Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = WarmupScheduler(optimizer=optimizer, warmup_steps=5, T_max=95, lr_min=1e-8)

    save_dir = args.save_dir
    log_file_path = os.path.join(save_dir, "log.txt")

    if not os.path.exists(log_file_path):
        with open(log_file_path, "a") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')

    args_file_path = os.path.join(save_dir, 'args.txt')
    if not os.path.exists(args_file_path):
        with open(args_file_path, "w") as f:
            f.write(str(args))

    best_miou = 0.0
    if args.load_model is not None:
        print(f"Loading init_model from {args.init_model}")
        checkpoint = torch.load(args.init_model)
        state_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(OrderedDict(pretrained_dict))

    for index in range(0, args.max_epoch):
        epoch = index + 1
        learning_rate = optimizer.param_groups[0]['lr']
        time.sleep(0.1)

        print(f"Current Epoch: {epoch}/{args.max_epoch}, lr: {learning_rate}")
        train_fn(args, model, train_loader, loss_fn, optimizer)
        scheduler.step()
        metric_dict = check_metric(args, val_loader, model)

        with open(log_file_path, "a") as f:
            f.write(f"epoch: {epoch}, learning rate: {learning_rate}, "
                    f"acc: {metric_dict['acc']:.4f}, iou: {metric_dict['iou']:.4f}, "
                    f"acc_class: {metric_dict['acc_class']:.4f}, mIoU: {metric_dict['mIoU']:.4f}, "
                    f"fwavacc: {metric_dict['fwavacc']:.4f}\n")

        # 保存miou最大的模型
        if best_miou < metric_dict['mIoU']:
            best_miou = metric_dict['mIoU']
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'mIoU': best_miou
            }, os.path.join(args.save_dir, "model_best.pth"))

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'mIoU': metric_dict['mIoU']
            }, os.path.join(args.save_dir, f"model_{epoch:06d}.pth"))


if __name__ == '__main__':
    set_random()
    parser = get_default_parser()
    parser.add_argument('--init_weights_path', type=str, default='./pretrained/mit_b4.pth', help='none')
    out_args = parser.parse_args()
    main(out_args)
