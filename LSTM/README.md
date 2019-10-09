# LSTM

![url](https://i.stack.imgur.com/b4sus.jpg)
https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras

**One-to-one**   
model.add(Dense(output_size, input_shape=input_shape))

**One-to-many** this option is not supported well as chaining models is not very easy in Keras, so the following version is the easiest one:   
model.add(RepeatVector(number_of_times, input_shape=input_shape))      
model.add(LSTM(output_size, return_sequences=True))      

**Many-to-one**   
model = Sequential()   
model.add(LSTM(1, input_shape=(timesteps, data_dim)))   

**Many-to-many** This is the easiest snippet when the length of the input and output matches the number of recurrent steps   
model = Sequential()   
model.add(LSTM(1, input_shape=(timesteps, data_dim), return_sequences=True))   
   
**Many-to-many when number of steps differ from input/output length**:   

                                        O O O  
                                        | | |  
                                  O O O O O O  
                                  | | | | | |    
                                  O O O O O O   
                                  
model = Sequential()   
model.add(LSTM(1, input_shape=(timesteps, data_dim), return_sequences=True))   
model.add(Lambda(lambda x: x[:, -N:, :]   

Where N is the number of last steps you want to cover (on image N = 3).   

From this point getting to:

                                        O O O
                                        | | |
                                  O O O O O O
                                  | | | 
                                  O O O 
                                  
is as simple as artificial padding sequence of length N using e.g. with 0 vectors, in order to adjust it to an appropriate size.

