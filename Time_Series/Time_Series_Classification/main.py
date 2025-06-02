import LSTM_Classification_Model
import Prepare_Data
import File_Preprocessor 



def main():
    num_classes = 17
    time_step = 5
    sampling_step = 1  
    train_ratio = 0.8
    input_path = "PID_1_BPM.csv"
    output_path = "PID_1_BPM_cleaned.csv"
    column_string = "sys;dia;hr;spo2;features"
    epochs = 128
    batch_size = 32


    processor = File_Preprocessor.CSVPreprocessor(input_path, output_path, column_string)
    processor.process()
    
    x_train, y_train, x_test, y_test = Prepare_Data.prepare_data(
        filepath = output_path,
        num_classes = num_classes,
        time_step = time_step,
        train_ratio = train_ratio,
        sampling_step = sampling_step 
    )

    input_shape = (x_train.shape[1], x_train.shape[2])

    model = LSTM_Classification_Model.LSTMClassifier(input_shape = input_shape, num_classes = num_classes)
    model.train(x_train = x_train, y_train = y_train, batch_size = batch_size, epochs = epochs)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")



if __name__ == "__main__":
    main()