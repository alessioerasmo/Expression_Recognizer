def plot_history(history):
    import matplotlib.pyplot as plt
    from tensorflow import keras
    print(history.history.keys())
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    """
    
   
    """