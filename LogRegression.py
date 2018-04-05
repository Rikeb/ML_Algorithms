class LogRegression(object):
    
    # Initial setup
    
    def __init__(self):
        
        self.weights = np.random.randn(x_train.shape[1] + 1, 1) # (N+1 x 1)
        self.weights = self.weights[1:]
        self.biases  = self.weights[0]
         
    # Log regrssion model development
    
    def activation(self, x_train):
        
        # Feedforward propogation 
        # Nonlinear neurons
        
    #    ones = np.zeros(x.shape[0],1)
        
      #  x = np.hstack((ones, x))
      #  y = np.dot(x, w)
        x = x_train
        a = [sigmoid(np.dot(x, w) + b) for w, b in zip(self.weights, self.biases)]
        
        return a
    
    # Fitting model -  parameter learning
    
    def SGD(self, epochs, batch = None, batch_size = None, alpha, x_train, y_train,
            test_data = None,
            monitor_test_cost = False,
            monintor_training_cost = False):
        
        n_test = len(test_data)
        test_cost, training_cost = [], []
        
        
        for j in range(epochs):
            random.shuffle(x_train)
            if batch == 'mini_batch':
                    batches = [x_tain[k : k + batch_size] for k in range(0,batch_size)]
                    for mini_batch in batches:
                        self.param_update(mini_batch, y_train, alpha)
            elif batch  == 'batch':
                    batches = [x_tain[k : k + batch_size] for k in range(0,batch_size)]
                    self.param_update(batches, y_train, alpha)
            else:
                self.param_update(x_train, y_train, alpha) # or pass mini_batches
            
            print("Epoch {} complete".format(j))   
                
            if monitor_test_cost:
                cost = self.cost(x_test, y_test)
                test_cost.append(cost)
                print("Epoch {}: Cost on test data: {}".format(j, cost))
            if monitor_training_cost:
                cost = self.cost(x_train, y_train)
                training_cost.append(cost)
                print("Epoch {}: Cost on training data: {}".format(j, cost))
            
         #   if test_data:
         #       print("Epoch {} : {} / {}".format(j, self.predict(test_data),n_test));
         #   else:
         #       print("Epoch {} complete".format(j))
           
        return evaluation_cost, training_cost, self.weights, self.biases
           
            
       
    def param_update(self, alpha, x_train, y_train, alpha):
        
        """Update the network's weights and biases by applying
        gradient descent learning"""
        n_samples = x_train.shape[0]
        y_predict = self.activation(x_train)
        error = y_train - output
        # weight and bias update
        dw = ((1/n_samples) * np.dot(x_train.T, (y_train - y_predict)))
        db = ((1/n_samples) * np.sum(y_train - y_predict)
        
        # memory allocation for nabla_w and nabla_b parameters
     #   nabla_b = [np.zeros(b.shape) for b in self.biases]
     #   nabla_w = [np.zeros(w.shape) for b in self.weights]              ]
        
        
        
     #   self.weights = [w - (alpha/n_samples) * dw for w in self.weights]
     #   self.biases  = [b - (alpha/n_samples) * db for b in self.biases]
        
     # return self.weights, self.biases
        
    
        """ def backpropogate(self, x_train, y_train) 
    
        # compute cost and cost derivative
    
        y = y_train
        a = self.feedforward(x_train)
        cost = np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))) """     
              
    def cost(self, x_train, y_train):
    
        a = self.activation(x_train)
        cost = (1/x_train.shape[1]) 
        return cost
#* np.sum(np.nan_to_num(-y_train*np.log(a)-(1-y_train)*np.log(1-a)))
    
    def predict(self, x):
        
        # Classify x as class 1 if activation > 0.5, else classify as class 0
        a = self.activation(x)
        y_predict_labels = np.where(a >= 0.5, 1,0)
        
        return y_predict_labels[:, newaxis]

    
#*******************************************************************
    
    
    def sigmoid(z):
        
        return 1/(1 + np.exp(-z))










