# Filename: utils.py
import numpy as np
import cv2

# Implements softmax function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def predictDigit(image,net):
    # Save image
    img = image[:,:,0:3]
    img = np.array(img, dtype=np.uint8)
    #print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.jpg",img)

    # Read image in grayscale mode
    img = cv2.imread("test.jpg",cv2.IMREAD_GRAYSCALE)

    # Create a 4D blob from image
    blob = cv2.dnn.blobFromImage(img, 1/255, (28, 28))

    # Run a model
    net.setInput(blob)
    out = net.forward()

    # Get a class with a highest score
    out = softmax(out.flatten())
    #print(out)
    classId = np.argmax(out)
    confidence = out[classId]

    return classId, confidence
