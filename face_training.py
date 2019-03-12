from fastai.vision import *

folder = 'human'
file = 'urls_human.txt'

#folder = 'paint'
#file = 'urls_paint.txt'

#folder = 'sculpture'
#file = 'urls_sculpture.txt'

path = Path('data/face')
dest = path/folder
#dest.mkdir(parents=True, exist_ok=True)

classes = ['human','paint', 'sculpture']

#download the images for one category
download_images(dest/file, dest, max_pics=200) 

#remove any images that can't be open
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)

#view data
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

#print classes
data.classes

#show some images
data.show_batch(rows=3, figsize=(7,8))

#train model
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()

#find best learning rate
learn.lr_find()
learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))
learn.save('stage-2')

#Clean up the data
from fastai.widgets import *
ds, idxs = DatasetFormatter().from_toplosses(learn, ds_type=DatasetType.Valid)
ImageCleaner(ds, idxs, path)

ds, idxs = DatasetFormatter().from_similars(learn, ds_type=DatasetType.Valid)
ImageCleaner(ds, idxs, path, duplicates=True)

#putting model in production
learn.export()
img = open_image(path/'test.jpg')
img
learn = load_learner(path)
pred_class,pred_idx,outputs = learn.predict(img)
pred_class





















