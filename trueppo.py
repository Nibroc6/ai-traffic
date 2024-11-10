# Evaluate the restored model
new_model = tf.keras.models.load_model('generic_model.keras')

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)