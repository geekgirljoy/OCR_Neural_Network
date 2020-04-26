<?php


ini_set("max_execution_time", "-1");
ini_set('memory_limit','-1');
set_time_limit(0);

include('Functions.php');


// Training Variables
$desired_error = 0.001; //0.025
$max_epochs = 500000;
$current_epoch = 0;
$epochs_between_saves = 5; // Minimum number of epochs between saves
$epochs_since_last_save = 0;

// Training Data
$name  = 'minst.ocr';
$path = __DIR__ . DIRECTORY_SEPARATOR . 'ANNs';
@mkdir($path, 0777);

$data =  __DIR__ . DIRECTORY_SEPARATOR . 'Training Data' . DIRECTORY_SEPARATOR . 'minst.train.data';

// Initialize pseudo mse (mean squared error) to a number greater than the desired_error
// this is what the network is trying to minimize.
$pseudo_mse_result = $desired_error * 10000; // 1
$best_mse = $pseudo_mse_result; // keep the last best seen MSE network score here

// Initialize ANN
$num_input = 392;
$num_output = 10;

$hidden_layers = array(1=>512 // First Hidden Layer - 256 neurons
                       // Add More layers as needed (don't forget the commas)
                      );

$layers = array($num_input, $hidden_layers, $num_output);
$layers = FlattenANNLayers($layers);
$num_layers = count($layers);

// Create ANN
$ann = fann_create_standard_array($num_layers, $layers);


if($ann){
  
  $log = fopen($path . DIRECTORY_SEPARATOR . "traning_save_log.$name.txt", 'w');

 
  // Configure the ANN
  fann_set_activation_function_hidden($ann, FANN_SIGMOID_SYMMETRIC); // FANN_SIGMOID_SYMMETRIC
  fann_set_activation_function_output($ann, FANN_SIGMOID_SYMMETRIC); // FANN_SIGMOID_SYMMETRIC
 
 echo 'Loading data from:'. $data . PHP_EOL;

  // Read training data
  $train_data = fann_read_train_from_file($data);
  
  echo 'Training ANN... '. $name . PHP_EOL;
  
  echo "Inputs: $num_input" . PHP_EOL;
  echo 'Hidden Layers: ' . count($hidden_layers) . PHP_EOL;
  foreach($hidden_layers as $hidden_layer=>$neuron_count){
      echo "H$hidden_layer: $neuron_count" . PHP_EOL;
  }
  echo "Outputs: $num_output" . PHP_EOL;
  echo str_repeat('-', 50) . PHP_EOL;
   
 
 
  // Check if pseudo_mse_result is greater than our desired_error
  // if so keep training so long as we are also under max_epochs
  while(($pseudo_mse_result > $desired_error) && ($current_epoch <= $max_epochs)){
      $current_epoch++;
      $epochs_since_last_save++; 
     
      // See: http://php.net/manual/en/function.fann-train-epoch.php
      // Train one epoch
      //
      // One epoch is where all of the training data is considered
      // exactly once.
      //
      // This function returns the MSE error as it is calculated
      // either before or during the actual training. This is not the
      // actual MSE after the training epoch, but since calculating this
      // will require to go through the entire training set once more.
      // It is more than adequate to use this value during training.
      $pseudo_mse_result = fann_train_epoch($ann, $train_data);
      
      echo "$name " . $current_epoch . ' : ' . $pseudo_mse_result . PHP_EOL; // report
       
      // If we haven't saved the ANN in a while...
      // and the current network is better then the previous best network
      // as defined by the current MSE being less than the last best MSE
      // Save it!
      if(($epochs_since_last_save >= $epochs_between_saves) && ($pseudo_mse_result < $best_mse)){
       
        $best_mse = $pseudo_mse_result; // we have a new best_mse
       
        // Save a Snapshot of the ANN
        fann_save($ann, $path . DIRECTORY_SEPARATOR . "$name.train.net");
        echo "Saved $name ANN." . PHP_EOL; // report the save
        $epochs_since_last_save = 0; // reset the count
        
        fwrite($log, $pseudo_mse_result . PHP_EOL);
      }
 
  } // While we're training

  echo 'Training Complete! Saving Final Network.'  . PHP_EOL;
 
  // Save the final network
  fann_save($ann, $path . DIRECTORY_SEPARATOR . "$name.final.net"); 
  fann_destroy($ann); // free memory
  fclose($log);
}
echo 'All Done!' . PHP_EOL;
?>