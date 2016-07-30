/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes

	public ArrayList<Instance> trainingSet=null;//the training set

	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs

	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. 
	 */

	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;

		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}

		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);

		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);

		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}

	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.5], it should return 2. 
	 * The parameter is a single instance. 
	 */	
	public int calculateOutputForInstance(Instance inst)
	{	
		// update inputValue
		for (int i = 0; i < inputNodes.size() - 1; i++) {
			inputNodes.get(i).setInput(inst.attributes.get(i));
		}
		// calculate output for hidden nodes
		//System.out.println("calculate output for hidden nodes");
		for (int i = 0; i < hiddenNodes.size(); i++) {
			hiddenNodes.get(i).calculateOutput();
		}
		// calculate output for output nodes
		//System.out.println("calculate output for output nodes");
		for (int i = 0; i < outputNodes.size(); i++) {
			outputNodes.get(i).calculateOutput();
		}
		// get best index from output nodes
		int maxIndex = 0;
		for (int i = 0; i < outputNodes.size() - 1; i++){
			double newnumber = outputNodes.get(i).getOutput();
			if (newnumber >= outputNodes.get(maxIndex).getOutput()){
				maxIndex = i;
			}
		}
		return maxIndex;	
	}


	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */	
	public void train()
	{
		int numEpoch = 0;
		while (numEpoch < maxEpoch) {
			System.out.println("epoch #: "+numEpoch);
			// initialize weights
			//System.out.println("initialize w: input-hidden");
			for (int i = 0; i < hiddenNodes.size() - 1; i++) {
				for (NodeWeightPair pn: hiddenNodes.get(i).parents) {
					//System.out.println("-");
					pn.weight = Math.random() * 0.1;
				}				
			}
			//System.out.println("initialize w: hidden-output");
			for (int i = 0; i < outputNodes.size(); i++) {
				for (NodeWeightPair pn: outputNodes.get(i).parents) {
					pn.weight = Math.random() * 0.1;
				}
			}
			
			// for each training set instance
			for (Instance inst : trainingSet) {
				// for each input except for bias, set output
				//System.out.println("set output: input");
				for (int i = 0; i < inputNodes.size() - 1; i++) { // except for bias node
					inputNodes.get(i).setInput(inst.attributes.get(i));
				}
				// for each hidden node except for bias, set output
				//System.out.println("set output: hidden");
				for (int i = 0; i < hiddenNodes.size() - 1; i++) { // except for bias node
					hiddenNodes.get(i).calculateOutput();
				}
				// for each output node calculate delta j
				List<Double> deltaJList = new ArrayList<>();
				//System.out.println("get delta: hidden-output");
				for (int i = 0; i < outputNodes.size(); i++) {
					outputNodes.get(i).calculateOutput();
					double deltaJ = reluGradient(outputNodes.get(i).getSum()) * (inst.classValues.get(0) - outputNodes.get(0).getOutput());
					deltaJList.add(deltaJ);
				}
				// for each hidden node calculate delta i
				List<Double> deltaIList = new ArrayList<>();
				//System.out.println("get delta: input-hidden");
				for (int i = 0; i < hiddenNodes.size(); i++) {
					double sum = 0;
					for (int j = 0; j < outputNodes.size(); j++) {
						sum = sum + outputNodes.get(j).parents.get(i).weight * deltaJList.get(j);
					}
					double deltaI = reluGradient(hiddenNodes.get(i).getSum()) * sum;
					deltaIList.add(deltaI);
				}
				// update weights
				//System.out.println("update w: input-hidden");
				for (int i = 0; i < hiddenNodes.size() - 1; i++) {
					for (NodeWeightPair pn: hiddenNodes.get(i).parents) {
						pn.weight = pn.weight + learningRate * pn.node.getOutput() * deltaIList.get(i);
					}
				}
				//System.out.println("update w: hidden-output");
				for (int i = 0; i < outputNodes.size() - 1; i++) {
					for (NodeWeightPair pn: outputNodes.get(i).parents) {
						pn.weight = pn.weight + learningRate * pn.node.getOutput() * deltaJList.get(i);
					}
				}
			}
			numEpoch++;
		}
	}
	
    private double reluGradient(double in) {
        if(in < 0) return 0;
        else return 1;
    }
	
}
