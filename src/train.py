import tensorflow as tf

@tf.function
def train_step(model, optimizer, loss, 
               inputs, neigh_list, labels):
# funtion to perform one training step
  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predict = model(inputs, neighList, training=True)

    # fidelity loss usin mse
    total_loss = loss(predict, labels)

  # compute the gradients of the total loss with respect to the trainable variables
  gradients = tape.gradient(total_loss, model.trainable_variables)
  # update the parameters of the network
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss