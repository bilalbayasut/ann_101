<?php
require('ann.php');
// prepare XOR traing data
// no_data => input1, input2, output
$data=array(
			0=>array(1,	1,	0),
			1=>array(0,	1,	1),
			2=>array(1,	0,	1),
			3=>array(1,	1,	0),
			4=>array(0,	0,	0)
			);


// prepare test
// no_data => input1, input2, output
$testData=array(
				0=>array(0,	0,	0),
				1=>array(0,	0,	0),
				2=>array(0,	1,	1),
				3=>array(0,	1,	1),
				4=>array(1,	0,	1),
				5=>array(1,	0,	1),
				6=>array(1,	1,	0),
				7=>array(1,	1,	0)
			);

$layer_size = array(2,3,1);
$num_of_layers = count($layer_size);

$learning_rate = 0.3;

$ann = new Ann($num_of_layers, $layer_size, $learning_rate);
$ann->run($data,$testData);

?>
