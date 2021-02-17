import tensorflow as tf
import tensorflow_probability as tfp
from tqdm.notebook import tqdm

from scripts.mixmatch import (ema, interleave, linear_rampup, mixmatch, semi_loss,
                      weight_decay)


def train(datasetX, datasetU, model, ema_model, optimizer, epoch, args):
    # performance measures
    xe_loss_avg = tf.keras.metrics.Mean()
    l2u_loss_avg = tf.keras.metrics.Mean()
    total_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.CategoricalAccuracy()

    # batch function
    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], drop_remainder=True)

    iteratorX = iter(shuffle_and_batch(datasetX))
    iteratorU = iter(shuffle_and_batch(datasetU))
    
    # create beta distribution to sample during mixup
    beta = tfp.distributions.Beta(args['alpha'], args['alpha'])

    # create good looking progress bar
    progress_bar = tqdm(range(args['val_iteration']), unit='batch')
    for batch_num in progress_bar:
        # trade-off factor between labelled loss & unlabelled loss. Unlabelled loss starts with low importance and slowly increases.
        lambda_u = tf.constant(args['lambda_u'] * linear_rampup(epoch + batch_num/args['val_iteration'], args['rampup_length']))

        # get next batch of unlabelled & labelled data. If there is no next batch -> reshuffle
        try:
            batchX = next(iteratorX)
        except:
            iteratorX = iter(shuffle_and_batch(datasetX))
            batchX = next(iteratorX)
        try:
            batchU = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(datasetU))
            batchU = next(iteratorU)

        # Gradient tape allows us to calculate gradients be remembering previous losses
        with tf.GradientTape() as tape:
            # run mixmatch
            XU, XUy = mixmatch(model, tf.cast(batchX['image'], tf.float32), tf.one_hot(batchX['label'], 10), tf.cast(batchU['image'], tf.float32), tf.constant(args['T']), args['K'], beta.sample())

            # obtain & restructure predictions for labelled (logits_x) and unlabelled (logits_u)
            logits = [model(batch) for batch in XU]
            logits = interleave(logits, args['batch_size'])
            logits_x = logits[0]
            logits_u = tf.concat(logits[1:], axis=0)

            # compute loss with trade-off factor
            xe_loss, l2u_loss = semi_loss(XUy[:args['batch_size']], logits_x, XUy[args['batch_size']:], logits_u)
            total_loss = xe_loss + lambda_u * l2u_loss


        # compute gradients and run optimizer step
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # update Exponential Moving Average model
        ema(model, ema_model, args['ema_decay'])

        # apply weight decay to model to prevent overfitting
        weight_decay(model=model, decay_rate=args['weight_decay'] * args['learning_rate'])

        # update performance measures
        xe_loss_avg.update_state(xe_loss)
        l2u_loss_avg.update_state(l2u_loss)
        total_loss_avg.update_state(total_loss)
        accuracy.update_state(tf.one_hot(batchX['label'], 10), model(tf.cast(batchX['image'], dtype=tf.float32)))

        # update progess bas stats
        progress_bar.set_postfix({
            'XE Loss': f'{xe_loss_avg.result():.4f}',
            'L2U Loss': f'{l2u_loss_avg.result():.4f}',
            'WeightU': f'{lambda_u:.3f}',
            'Total Loss': f'{total_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}'
        })


def validate(dataset, model, epoch, args, split):
    # performance metrics
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    xe_avg = tf.keras.metrics.Mean()


    for batch in dataset.batch(args['batch_size']):
        # predict
        logits = model(tf.cast(batch['image'], dtype=tf.float32), training=False)

        # calculate loss
        xe_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(batch['label'], 10), logits=logits))

        # update performance measures
        xe_avg.update_state(xe_loss)
        accuracy.update_state(tf.one_hot(batch['label'], 10), logits)

    # print stats
    print(f'Epoch {epoch:04d}: {split} XE Loss: {xe_avg.result():.4f}, {split} Accuracy: {accuracy.result():.3%}')
