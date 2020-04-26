<?php


ini_set("max_execution_time", "-1");
ini_set('memory_limit','-1');
set_time_limit(0);

//include('Functions.php');



$path = __DIR__ . DIRECTORY_SEPARATOR . 'ANNs' . DIRECTORY_SEPARATOR ;
$test_data =  __DIR__ . DIRECTORY_SEPARATOR . 'Training Data' . DIRECTORY_SEPARATOR . 'minst.test.data';


// CSV to log test results
$test_results_csv = fopen($path . 'results.csv', 'w');

// Load ANN
$ann_train_file = ($path . "minst.ocr.final.net"); // 94.08% Accuracy on test data
//$ann_train_file = ($path . "minst.ocr.train.net");   // Used for testing durining training or if a final was not saved


if (!is_file($ann_train_file)){
    die("The .net file has not been created!" . PHP_EOL);
}

$ann = fann_create_from_file($ann_train_file);

if ($ann) {
    
    // Some variable to keep track of things
    $current_input = '';
    $current_output = '';
    $current_line = 0;
    
    // Open the test data file 
    $test_file = fopen($test_data, "r"); 
    fputcsv($test_results_csv, array('Ann Answer', 
                                     'Correct Answer',
                                     'Answered Correctly',
                                     'Raw Sum',
                                     'Ideal Sum',
                                     'Variance',
                                     'Ideal Output', 
                                     'Raw Output'));


    $temp_correct_score = 0;
    
    
    // While we have not reached the end of the test data set
    while(!feof($test_file))
    {
        $data = str_replace(array(PHP_EOL, "\n", "\r"), '', fgets($test_file)); // Remove those pesky end of lines
        
        
        // If there remains data after removing the EOL
        if($data != ''){
            
            //////////////////////////////////////
            // What Type of Data is this?       //
            //////////////////////////////////////
            
            // If this is the first line in a FANN data file
            // then...
            if($current_line == 0){ // data is the header
                $type = 'Header';
            }
            // Otherwise if we can divide the current line number by two
            // and the result isn't zero... 
            elseif($current_line % 2 != 0){ // data is an input
                $type = 'Input';
                $current_input = $data;
            }
            // Otherwise the result was zero meaning that
            // this is of course...
            else{// an output
                $type = 'Output';
                $current_output = $data;
            }
            
            //////////////////////////////////////
            // If we have a complete data pair  //
            //////////////////////////////////////
            if($current_input != '' && $current_output != ''){        
            
                // Convert input string to array by using spaces as delimiters
                $input = explode(' ', $current_input);
                
                // ANN Calc inputs and store outputs in the result array
                $result = fann_run($ann, $input);
                
                // There are 10 outputs representing 
                // 0 - 9
                // [0,1,2,3,4,5,6,7,8,9]
                //
                // Which output contains the highest value? (the prediction/classification)
                $calc_digit = max($result); 
                
                // Look up the position of the Highest value in the array
                // it's key is the selection, in this case the actual digit
                // but it could be cat/no cat, dog/cat, red/green/blue,
                // lat/long, credit history, image contains licence place yes/no,
                // etc... whatever "classification" you assign the output to mean.
                // as long as there is corilation between inputs and outputs... this
                // should gennerally hold true so long as you process and train your 
                // model properly, though some systems can be increadably complex
                // requireing multiple layers of processing and stacked network layers
                // which are the so called "deep" neural networks.
                
                $ann_answer = array_search($calc_digit, $result);// The ANN answer
                $raw_sum = array_sum($result);
                $raw_output = implode(' ', $result);
                
                // The correct answer is:
                $ideal_output = $current_output;
                $current_output = explode(' ', $current_output);
                $calc_digit = max($current_output);
                $correct_answer =  array_search($calc_digit, $current_output);
                $ideal_sum = array_sum($current_output);
                
                // Did the ANN answer correctly?
                $answered_correctly = -1;
                if($ann_answer == $correct_answer){
                    $answered_correctly = 1;
                }
                
                // Very roughly how far off were all the answers
                $variance = $ideal_sum - $raw_sum;
                
                // Log results to CSV for data science
                // happy fun times later!                    
                fputcsv($test_results_csv, array($ann_answer, 
                                                 $correct_answer,
                                                 $answered_correctly,
                                                 $raw_sum,
                                                 $ideal_sum,
                                                 $variance,
                                                 $ideal_output, 
                                                 $raw_output));
                                                 
                if($answered_correctly == 1){
                    $temp_correct_score+= $answered_correctly;
                }
                
                // Reset input and output set
                $current_input = '';
                $current_output = '';
            }
            $current_line++; // Next line
        }
    }
    fclose($test_file); // Close Test file
    
    echo 'Number Correct: ' . $temp_correct_score / 100 . '%' . PHP_EOL;

    // Forceably remove the neural network from this plane of existance
    fann_destroy($ann); 

    fclose($test_results_csv); // Close CSV
    
}else{
    die("Invalid file format" . PHP_EOL);
}



//////////////////////
// Happy Fun Times! //
//////////////////////

$errors_by_number = array(0=>0, 1=>0, 2=>0, 3=>0, 4=>0, 5=>0, 6=>0, 7=>0, 8=>0, 9=>0);

$n = 0;
if (($test_results_csv = fopen($path . "results.csv", "r")) !== FALSE){ // If we can read the results .csv file
    while (($results = fgetcsv($test_results_csv)) !== FALSE) {
        
        if($n > 0){

            /*
            $results[i] keys:
            
            key 0 = ann_answer, 
            key 1 = correct_answer,
            key 2 = answered_correctly,
            key 3 = raw_sum,
            key 4 = ideal_sum,
            key 5 = variance,
            key 6 = ideal_output, 
            key 7 = raw_output
            */
            // Check if ANN answer matched the correct answer
            if($results[0] != $results[1]){
                // What wrong answer (which number) was given
                // Log it
                $errors_by_number[$results[0]] += 1;
            }
        }
        
        $n++;
    }

    fclose($test_results_csv);

}

$errors_by_number_csv = fopen($path . 'errors_by_number.csv', 'w');
$results = fputcsv($errors_by_number_csv, array('0','1','2','3','4','5','6','7','8','9'));
$results = fputcsv($errors_by_number_csv, $errors_by_number);
fclose($errors_by_number_csv);



echo 'All Done!';
