<?php
define("_RAND_MAX",32767);
class Ann{

public $num_of_layers=0;
public $layer_size=0;
public $learning_rate=0;
public $new_weights=null;
public $weights = array();
public $delta=array();
public $output=array();
public $data=array();
public $testData = array();


public function __construct($num_of_layers, $layer_size, $learning_rate) {

  	$this->num_of_layers = $num_of_layers;
  	$this->layer_size = $layer_size;
  	$this->learning_rate = $learning_rate;

  	//seed the weights of the layers
  	for($i=1;$i<$this->num_of_layers;$i++){
  		for($j=0;$j<$this->layer_size[$i];$j++){
  			for($k=0;$k<$this->layer_size[$i-1]+1;$k++)
			{
				$this->weights[$i][$j][$k]=$this->random();
			}
			// bias in the last neuron
			$this->weights[$i][$j][$this->layer_size[$i-1]]=-1;
  		}
  	}


  }

public function forward($inputSource){

  	$sum = 0.0;
  	//assign content to input layer
  	for($i=0;$i<$this->layer_size[0];$i++){
  		$this->output[0][$i] = $inputSource[$i];
  	}

  		//assign sum and activation function to each neuron
  		for($i=1;$i<$this->num_of_layers;$i++)
		{
			for($j=0;$j<$this->layer_size[$i];$j++)
			{
				$sum=0.0;
				for($k=0;$k<$this->layer_size[$i-1];$k++)
				{
	                $sum+=$this->output[$i-1][$k]*$this->weights[$i][$j][$k];  // Apply weight to inputs and add to sum
				}
				// Apply bias
				$sum+=$this->weights[$i][$j][$this->layer_size[$i-1]];
				// Apply sigmoid function
				$this->output[$i][$j]=$this->sigmoid($sum);
			}
		}



  }

public function backward($inputSource,$target){

  $this->forward($inputSource);


	// FIND DELTA FOR OUPUT LAYER (Last Layer)

	for($i=0;$i<$this->layer_size[$this->num_of_layers-1];$i++)
	{	//\delta_{o1} = out_{o1}(1 - out_{o1}) * -(target_{o1} - out_{o1})
		$this->delta[$this->num_of_layers-1][$i]=$this->output[$this->num_of_layers-1][$i]*(1-$this->output[$this->num_of_layers-1][$i])*($target-$this->output[$this->num_of_layers-1][$i]);
	}


	//FIND DELTA FOR HIDDEN LAYERS (From Last Hidden Layer BACKWARDS To First Hidden Layer)

	for($i=$this->num_of_layers-2;$i>0;$i--)
	{
		for($j=0;$j<$this->layer_size[$i];$j++)
		{
			$sum=0.0;
			for($k=0;$k<$this->layer_size[$i+1];$k++)
			{
				$sum+=$this->delta[$i+1][$k]*$this->weights[$i+1][$k][$j];
			}
			$this->delta[$i][$j]=$this->output[$i][$j]*(1-$this->output[$i][$j])*$sum;
		}
	}

	// $this->debug();

	// ADJUST WEIGHT
	for($i=1;$i<$this->num_of_layers;$i++)
	{
		for($j=0;$j<$this->layer_size[$i];$j++)
		{
			for($k=0;$k<$this->layer_size[$i-1];$k++)
			{

				$this->new_weights[$i][$j][$k]=$this->learning_rate*$this->delta[$i][$j]*$this->output[$i-1][$k];
				$this->weights[$i][$j][$k]+=$this->new_weights[$i][$j][$k];
			}

			/* --- Apply the corrections */
			$this->new_weights[$i][$j][$this->layer_size[$i-1]]=$this->learning_rate*$this->delta[$i][$j];
			$this->weights[$i][$j][$this->layer_size[$i-1]]+=$this->new_weights[$i][$j][$this->layer_size[$i-1]];
		}
	}

  }

    public function debug(){
    echo "<br/>";
    echo " weights :";
    echo "<br/>";
    print_r($this->weights);
    echo "<br/>";
    echo "<br/>";
    echo " outputs :";
    echo "<br/>";
    print_r($this->output);
    echo "<br/>";
     echo "<br/>";
    echo " delta :";
    echo "<br/>";
    print_r($this->delta);
    echo "<br/>";
      echo "<br/>";
    echo " new_weights :";
    echo "<br/>";
    print_r($this->new_weights);
    echo "<br/>";
    die();
  }

  protected function sigmoid($inputSource){
  	return abs((double)(1.0 / (1.0 + exp(-$inputSource))));
  }

  protected function random(){
  	return (double)(rand())/(_RAND_MAX/2) - 1;//32767
  }

  protected function sigmoidPrime($value){
  	return exp($value)/(pow((1+exp($value)),2));

  }


  protected function mse($target){
  	$mse=0;

	for($i=0;$i<$this->layer_size[$this->num_of_layers-1];$i++)
	{
		$mse+=($target-$this->output[$this->num_of_layers-1][$i])*($target-$this->output[$this->num_of_layers-1][$i]);
	}
	return $mse/2;
  }

  // returns i'th outputput of the net

public function Out($i){
	return $this->output[$this->num_of_layers-1][$i];
}

public function run($data,$testData){

  	/* --- Threshhold - thresh (value of target mse, training stops once it is achieved) */
	$Thresh =  0.0001;
	$numEpoch = 200000;
	$MSE=0.0;
	$NumPattern=count($data);	// Lines
	$NumInput=count($data[0]);	// Columns

	echo  "\nNow training the network.... <br/>";

	for($e=0;$e<$numEpoch;$e++)
	{
		/* -- Backpropagate */
		$this->backward($data[$e%$NumPattern],$data[$e%$NumPattern][$NumInput-1]);

		$MSE=$this->mse($data[$e%$NumPattern][$NumInput-1]);
		if($e==0)
		{
			echo "\nFirst epoch Mean Square Error: $MSE <br/>";
		}

		if( $MSE < $Thresh)
		{
           echo "\nNetwork Trained. Threshold value achieved in ".$e." iterations. <br/>";
           echo "\nMSE:  ".$MSE. "<br/>";
           break;
        }
	}

	echo "\nLast epoch Mean Square Error: $MSE <br/>";
	echo "<br/>";
	echo "\nNow using the trained network to make estimations on test data.... <br/>";
	echo "<table>";
	echo "<th> X1 </th>";
	echo "<th> X2 </th>";
	echo "<th> Estimation </th>";

    for ($i = 0 ; $i < count($testData); $i++ )
    {

        $this->forward($testData[$i]);

        echo "\n";
        echo "<tr>";

		for($j=0;$j<$NumInput-1;$j++)
		{
			echo "<td>".$testData[$i][$j]."</td>";
		}

		echo "<td>".(double)$this->Out(0)."</td>";
		echo "</tr>";
    }
    echo "</table>";


  }

}

?>
