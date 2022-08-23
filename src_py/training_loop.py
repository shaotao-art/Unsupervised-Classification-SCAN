import torch
import time
import numpy as np

def simclr_train(train_loader, model, criterion, optimizer, epoch, device, logger):
    """
    train simclr
    """
    model.train()
    start = time.time()
    loss_lst = []
    for i, (x) in enumerate(train_loader):
        images = x[0]
        images_augmented = x[1]
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.to(device)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_lst.append(loss.item())
        if i % (len(train_loader) // 10) == 0:
            now = time.time()
            time_used = int(now - start)
            logger.info(f'epoch:[{epoch}], batch:[{i}/{len(train_loader)}], time used: {time_used // 60} min {time_used % 60} sec, loss:[{loss:>4f}]')

    end = time.time()
    time_used = int(end - start)
    logger.info(f'epoch:[{epoch}], avg loss: {np.array(loss_lst).mean()}, time to run this epoch: {time_used // 60} min {time_used % 60} sec')


def scan_train(train_loader, model, criterion, optimizer, epoch, device, logger, update_cluster_head_only):
    """ 
    Train w/ SCAN-Loss
    """

    model.train() # Update BN
    start = time.time()
    total_loss_lst = []
    consistency_loss_lst = []
    entropy_loss_lst = []
    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].to(device)
        neighbors = batch['neighbor'].to(device)
      
        # Calculate gradient for backprop of complete network

        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model.backbone(anchors)
                neighbors_features = model.backbone(neighbors)
            anchors_output = model.head(anchors_features)
            neighbors_output = model.head(neighbors_features)

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)    
        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        tmp1 = np.mean([v.item() for v in total_loss])
        tmp2 = np.mean([v.item() for v in consistency_loss])
        tmp3 = np.mean([v.item() for v in entropy_loss])
        total_loss_lst.append(tmp1)
        consistency_loss_lst.append(tmp2)
        entropy_loss_lst.append(tmp3)


        if i % (len(train_loader) // 10) == 0:
            now = time.time()
            time_used = int(now - start)
            logger.info(f'epoch:[{epoch}], batch:[{i}/{len(train_loader)}], time used: {time_used // 60} min {time_used % 60} sec, loss:[total:{tmp1:>4f}, consistency:{tmp2:>4f}, entropy:{tmp3:>4f}]')

    end = time.time()
    time_used = int(end - start)
    logger.info(f'epoch:[{epoch}], avg loss: total:{np.array(total_loss_lst).mean():>4f}, consistency:{np.array(consistency_loss_lst).mean():>4f}, entropy:{np.array(entropy_loss_lst).mean():>4f}, time to run this epoch: {time_used // 60} min {time_used % 60} sec')



def selflabel_train(train_loader, model, criterion, optimizer, epoch, device, ema, logger):
    """ 
    Self-labeling based on confident samples
    """
    model.train()
    start = time.time()
    loss_lst = []
    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        images_augmented = batch['image_augmented'].to(device)

        with torch.no_grad(): 
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        loss_lst.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)

        if i % (len(train_loader) // 10) == 0:
            now = time.time()
            time_used = int(now - start)
            logger.info(f'epoch:[{epoch}], batch:[{i}/{len(train_loader)}], time used: {time_used // 60} min {time_used % 60} sec, loss:[{loss:>4f}]')

    end = time.time()
    time_used = int(end - start)
    logger.info(f'epoch:[{epoch}], avg loss: {np.array(loss_lst).mean()}, time to run this epoch: {time_used // 60} min {time_used % 60} sec')

