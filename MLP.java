import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MLP {
	private static final int d = 2;   //# of inputs
	private static final int k = 3;   //# of categories
	private static final int H1 = 8; //# of 1st hidden layer neurons
	private static final int H2 = 4;  //# of 2nd hidden layer neurons
	private static final int H3 = 8; //# of 3rd hidden layer neurons

	private static final int size = 4000;
	private static final int batchSize = 2;
	private static List<String[]> trainContents = new ArrayList<String[]>(size); 
	private static List<String[]> testContents = new ArrayList<String[]>(size); 
	
	private static final int epochs = 700; 
	private static final float learningRate = 0.002f;
	private static final float terminationThreshold = 0.01f;
	
	private static final String activationFunctionH1 = "logistic"; 
	private static final String activationFunctionH2 = "tanh"; 
	private static final String activationFunctionH3 = "tanh"; 
	private static final String activationFunctionOutput = "relu"; 

	private static int nLayers = 5;
	private static Layer[][] layers; //1st D for layer, 2nd D for neurons -> layers[1][4] means 1st layer's 4th neuron
	
	private static float[][] desiredOutputTrain = new float[size][k];
	private static float[][] inputDataTrain = new float[size][d];
	private static float[][] desiredOutputTest = new float[size][k];
	private static float[][] inputDataTest = new float[size][d];
		
	public static void readFile(String fileName, List<String[]> list)
	{
		BufferedReader reader;

		try {
			reader = new BufferedReader(new FileReader(fileName));
			String line = reader.readLine();

			while (line != null) {
				String[] contents = line.split(","); 
				if(contents.length == 3)
					list.add(contents);

				line = reader.readLine();
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private static void initMemory()
	{
		layers = new Layer[nLayers][];
		layers[0] = new Layer[d]; //initialize sizes
		layers[1] = new Layer[H1];
		layers[2] = new Layer[H2];
		layers[3] = new Layer[H3];
		layers[4] = new Layer[k]; 

		for(int i = 0; i < nLayers;i++)
			for(int j = 0; j < layers[i].length; j++)
			{
				if(i>=1)
					layers[i][j] = new Layer(layers[i-1].length);
				else
					layers[i][j] = new Layer(layers[i].length);
			}
		
		for(int i = 0; i < nLayers; i++)
			for(int j = 0; j < layers[i].length; j++)
			{
				if(i == 1) layers[i][j].function = activationFunctionH1;
				if(i == 2) layers[i][j].function = activationFunctionH2;
				if(i == 3) layers[i][j].function = activationFunctionH3;
				if(i == 4) layers[i][j].function = activationFunctionOutput;
			}
		
	}
	
	private static void gradients(int currLayer, float[] output, float[] desiredO)
	{
		float nValue, derivative = 0, gradient = 0;
		for(int n = 0; n < layers[currLayer].length; n++)
		{
			nValue = layers[currLayer][n].nValue;
			
			if(currLayer == 1)
				derivative = derivativeActivationFunction(activationFunctionH1, nValue);
			else if(currLayer == 2)
				derivative = derivativeActivationFunction(activationFunctionH2, nValue);
			else if(currLayer == 3)
				derivative = derivativeActivationFunction(activationFunctionH3, nValue);
			else if(currLayer == 4)
				derivative = derivativeActivationFunction(activationFunctionOutput, nValue);
			else
				derivative = nValue * (1-nValue); 
			
			if(currLayer == nLayers-1)
				layers[currLayer][n].gradient = derivative*(output[n]-desiredO[n]);
			else
			{
				for(int h = 0; h <layers[currLayer+1].length; h++)
					gradient += layers[currLayer+1][h].gradient*layers[currLayer+1][h].weight[n];
				layers[currLayer][n].gradient = derivative*gradient;
			}
		}


	}

	public static float activationFunction(String func, float x)
	{
		
		float result = 0;
		if(func.equals("logistic"))
			result = (float) (1/(1+Math.exp(-x)));
		else if(func.equals("relu"))
			result = x > 0 ? x : 0;
		else if(func.equals("tanh"))
			result = (float)Math.tanh(x);
		
		return result;
	}
	
	public static float derivativeActivationFunction(String func, float x)
	{
		float result = 0;
		if(func.equals("logistic"))
			result = x*(1-x);
		else if(func.equals("relu"))
			result = x < 0 ? 0 : 1;
		else if(func.equals("tanh"))
			result = (float) (1 - Math.pow(x,2));
		
		return result;	
	}
	
	public static float MSE(float[] output, float[] desiredO)
	{
		float result = 0;
		float difference = 0;
		for(int i = 0; i < output.length; i++)
		{
			difference = output[i]-desiredO[i];
			result += 0.5*Math.pow(difference, 2);
		}
		return result;
	}
	
	public static float dotProduct(int length, Layer layers[], float[] weights)
	{
		float dotProduct = 0;
		for(int i = 0; i < length; i++)
			dotProduct += layers[i].nValue*weights[i];
		return dotProduct;
	}
	
	public static float[] feedForward(float[] input)
	{
		layers[0][0].nValue = input[0];
		layers[0][1].nValue = input[1];
		float[] output = new float[k];
		float bias, dotProduct = 0;
		for(int i = 1; i < nLayers; i++) 
			for(int j = 0; j < layers[i].length; j++) 
			{
				bias = layers[i][j].bias;
				dotProduct = dotProduct(layers[i-1].length, layers[i-1], layers[i][j].weight);
				dotProduct += bias;
				layers[i][j].dotProduct = dotProduct;
				layers[i][j].nValue = activationFunction(layers[i][j].function, dotProduct);
				if(i == nLayers - 1) 
					output[j] = layers[i][j].nValue;
				
			}
		return output;
	}
	
	public static void backPropagation(float[] output, float[] desiredO)
	{
		for(int i = nLayers-1; i > 0; i--)
			for(int j = 0; j <layers[i].length; j++)
			{
				gradients(i, output, desiredO);
				for(int l = 0; l < layers[i-1].length; l++)
				{
					layers[i][j].errors[l] += layers[i][j].gradient*layers[i-1][l].nValue;
					layers[i][j].errBias   += layers[i][j].gradient;
				}
			}
	}
	
	public static void train()
	{
		int epoch = 0;
		float errorDiff = 999999999, prevE = 999999999;
		float totalError = 0;
		float[] output = new float[k];
			
		while(epoch < epochs || errorDiff > terminationThreshold)
		{
			totalError = 0;
			for(int s = 0; s < size; s++)
			{
				output = feedForward(inputDataTrain[s]);
				backPropagation(output, desiredOutputTrain[s]);
				
				if((s+1)/batchSize != s/batchSize)
					for(int i = 1; i < nLayers; i++)
						for(int j = 0; j <layers[i].length; j++)
							for(int l = 0; l < layers[i-1].length; l++)
							{	//updating weights
								layers[i][j].weight[l] -= learningRate*(layers[i][j].errors[l]/batchSize);
								layers[i][j].bias -= learningRate*(layers[i][j].errBias/batchSize);
								layers[i][j].errors[l] = 0;
								layers[i][j].errBias = 0;
							}
				totalError += MSE(output, desiredOutputTrain[s]);
			}
			System.out.println("epoch: "+epoch+", total error: "+totalError);

			epoch++;
			errorDiff = Math.abs(prevE - totalError);
			prevE = totalError;
		}
	}
		
	public static void dataHandling()
	{
		for(int i = 0; i < trainContents.size(); i++) 
		{
			String[] contentsOfInputTrain = Arrays.toString(trainContents.get(i)).split(",");	
			inputDataTrain[i][0] = Float.parseFloat(contentsOfInputTrain[0].substring(1));
			inputDataTrain[i][1] = Float.parseFloat(contentsOfInputTrain[1]);
			String categoryTrain = contentsOfInputTrain[2].substring(1, contentsOfInputTrain[2].length() - 1);
			switch(categoryTrain) {
			case "1":
				desiredOutputTrain[i] = new float[] {1,0,0};
				break;
			case "2":
				desiredOutputTrain[i] = new float[] {0,1,0};
				break;
			case "3":
				desiredOutputTrain[i] = new float[] {0,0,1};
				break;
			}
			
			//accessing testing's contents
			String[] contentsOfInputTest = Arrays.toString(testContents.get(i)).split(",");	
			inputDataTest[i][0] = Float.parseFloat(contentsOfInputTest[0].substring(1));
			inputDataTest[i][1] = Float.parseFloat(contentsOfInputTest[1]);
			String categoryTest = contentsOfInputTest[2].substring(1, contentsOfInputTest[2].length() - 1);
			switch(categoryTest) {
			case "1":
				desiredOutputTest[i] = new float[] {1,0,0};
				break;
			case "2":
				desiredOutputTest[i] = new float[] {0,1,0};
				break;
			case "3":
				desiredOutputTest[i] = new float[] {0,0,1};
				break;
			}
		}
	}
	
	public static int maximize(float[] output)
	{
		int maxAt = 0;

		for (int i = 0; i < output.length; i++) 
		    maxAt = output[i] > output[maxAt] ? i : maxAt;
		return maxAt;
	}
	
	public static void evaluate()
	{
		float[] out;
		int correct = 0;
		for(int i = 0; i < inputDataTest.length; i++)
		{
			out = feedForward(inputDataTest[i]);
			int indexRes = maximize(out);
			int indexDesRes = maximize(desiredOutputTest[i]);
			if(indexRes == indexDesRes)
			{
				correct++; //hit
			}
		}
		System.out.println(correct + " correct out of " + size);
		System.out.println("Hit percentage: "+(float)correct/size);
	}
		
	public static void main(String[] args)
	{
		readFile("training.txt", trainContents);
		readFile("testing.txt", testContents);
		dataHandling();
		initMemory();
		train();
		evaluate();
	}
}


class Layer {
	private static final int min = -1;
	private static final int max = 1;

	float weight[], errors[];
	float gradient, bias, errBias, dotProduct, nValue;
	String function = ""; 
	
	Layer(int length)
	{
		weight = new float[length];
		errors = new float[length];
		for(int i = 0; i < length; i++)
			weight[i] = (float) (min + Math.random() * (max - min));
		bias = (float) (min + Math.random() * (max - min));
	}
}