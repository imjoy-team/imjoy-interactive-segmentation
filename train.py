import os
import tensorflow as tf
import segmentation_models as sm
import random

def trainGenerator(sample_pool):
    augmenter = A.Compose([
                       A.RandomCrop(362, 362),
                       A.OneOf([
                                A.VerticalFlip(), 
                                A.HorizontalFlip(),
                                A.RandomBrightnessContrast(p=0.8),    
                                A.RandomGamma(p=0.8)
                              ], p=1), 
                       A.RandomRotate90(p=1),
                       A.CenterCrop(256,256)
                     ])
    while True:
        x, y = get_one_sample(sample_pool)
        augmented = augmenter(image=x, mask=y)
        aug_x = augmented['image']
        aug_y = augmented['mask']
        yield (aug_x, aug_y)

def testGenerator(sample_pool):
    while True:
        x = get_one_sample(sample_pool)
        yield x

def train(batch_size=5,iterations=100):
    
    BACKBONE = 'mobilenetv2'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    sample_pool_train = load_sample_pool('data/hpa_dataset', folder='train')
    gen = trainGenerator(sample_pool_train)
    
    # define model
    model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=3, activation='sigmoid',
                    layers=tf.keras.layers, models=tf.keras.models, backend=tf.keras.backend, utils=tf.keras.utils)
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    
    #load previous weights TODO!
    #model = tf.keras.models.load_model('current_model.h5')
    
    #train
    print('Start training')
    for i in range(iterations):
        batchX = []
        batchY = []
        for i in range(batch_size):
            aug_im, aug_mask = next(gen)
            batchX += [aug_im]
            batchY += [aug_mask]

        model.train_on_batch(np.asarray(batchX), np.asarray(batchY))
    print(f'Finished {iterations} with batch size {batch_size}')
    model.save('current_model.h5')


def predict(img, size_limit=200):
    model = tf.keras.models.load_model('current_model.h5', compile=False)
    #img = load_image('data/hpa_dataset/test/sample1', img_type='image', resize=None)
    pred = model.predict(np.expand_dims(img, axis=0))

    pred[pred>0.7]=1
    pred[pred<1] = 0
    pred = pred.squeeze().astype('uint8')
    pred[:,:,2] = np.sum(pred, axis=2)
    pred[:,:,0] = 0
    pred[:,:,1] = 0
    pred = pred > 0
    pred = measure.label(pred[:,:,2]).astype(np.uint16)
    pred = morphology.remove_small_objects(pred, size_limit)
    pred = measure.label(pred).astype(np.uint16)
    return pred #mask_to_geojson(pred)