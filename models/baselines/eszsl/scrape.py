"""
Scrapes everything
"""
from data.imsitu_loader import ImSitu, CudaDataLoader, collate_fn
import numpy as np
from lib.imsitu_model import ImsituModel
from tqdm import tqdm
from lib.attribute_loss import AttributeLoss
import scipy.io as sio

train_data, val_data, test_data = ImSitu.splits(zeroshot=True)

for o in (train_data, val_data, test_data):
    o.iter = CudaDataLoader(dataset=o, batch_size=32*16, shuffle=False, num_workers=2,
                            collate_fn=collate_fn, volatile=True)

att_crit = AttributeLoss(train_data.attributes.domains, size_average=True)
# m = resnet152(pretrained=True)
m = ImsituModel(
    zeroshot=False,
    num_train_classes=1
)
m.load_pretrained(
    '/home/rowan/code/verb-attributes/checkpoints/imsitu_pretrain/pretrained_ckpt.tar'
)
m.cuda()
m.eval()

def extract(x):
    x = m.resnet152.conv1(x)
    x = m.resnet152.bn1(x)
    x = m.resnet152.relu(x)
    x = m.resnet152.maxpool(x)

    x = m.resnet152.layer1(x)
    x = m.resnet152.layer2(x)
    x = m.resnet152.layer3(x)
    x = m.resnet152.layer4(x)

    x = m.resnet152.avgpool(x)
    return x.view(x.size(0), -1).data.cpu().numpy()

train_feats = []
val_feats = []
test_feats = []
train_labels = []
val_labels = []
test_labels = []

for img_batch, label_batch in tqdm(val_data.iter):
    val_feats.append(extract(img_batch))
    val_labels.append(label_batch.cpu().data.numpy())

for img_batch, label_batch in tqdm(test_data.iter):
    test_feats.append(extract(img_batch))
    test_labels.append(label_batch.cpu().data.numpy())

for img_batch, label_batch in tqdm(train_data.iter):
    train_feats.append(extract(img_batch))
    train_labels.append(label_batch.cpu().data.numpy())

train_feats = np.concatenate(train_feats,0)
train_labels = np.concatenate(train_labels,0)
test_feats = np.concatenate(test_feats,0)
test_labels = np.concatenate(test_labels,0)
val_feats = np.concatenate(val_feats,0)
val_labels = np.concatenate(val_labels,0)


np.save('train_feats', train_feats)
np.save('test_feats', test_feats)
np.save('val_feats', val_feats)
np.save('train_labels', train_labels)
np.save('test_labels', test_labels)
np.save('val_labels', val_labels)


names = list(train_data.attributes.atts_df.index) + list(test_data.attributes.atts_df.index)

sio.savemat('imgdata.mat', {'X_train': train_feats,
                             'X_test': test_feats,
                              'X_val': val_feats,
                             'Y_train': train_labels,
                             'Y_test': test_labels,
                             'Y_val': val_labels,
                             'vocab': names
                             })