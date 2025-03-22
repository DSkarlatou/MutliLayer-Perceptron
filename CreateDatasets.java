import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.*;

public class CreateDatasets {
	private static   int min = -1;
	private static   int max = 1;

	public static float generate()
	{
		return (float) (min + Math.random() * (max - min));
	}

	public static void main(String[] args) throws IOException {
		
		FileWriter trainingDataset = new FileWriter("training.txt");	
		FileWriter testingDataset = new FileWriter("testing.txt");	
	    		
		for(int i = 0; i < 8000; i++)
		{
			float x1 = generate();
			float x2 = generate();
			int classification;
			
			if((x1-0.5)*(x1-0.5) + (x2-0.5)*(x2-0.5)<0.2 && x2>0.5)
				classification = 1;
			else if((x1-0.5)*(x1-0.5) + (x2-0.5)*(x2-0.5)<0.2 && x2<0.5)
				classification = 2;
			else if((x1+0.5)*(x1+0.5) + (x2+0.5)*(x2+0.5)<0.2 && x2>-0.5)
				classification = 1;
			else if((x1+0.5)*(x1+0.5) + (x2+0.5)*(x2+0.5)<0.2 && x2<-0.5)
				classification = 2;	
			else if((x1-0.5)*(x1-0.5) + (x2+0.5)*(x2+0.5)<0.2 && x2>-0.5)
				classification = 1;
			else if((x1-0.5)*(x1-0.5) + (x2+0.5)*(x2+0.5)<0.2 && x2<-0.5)
				classification = 2;
			else if((x1+0.5)*(x1+0.5) + (x2-0.5)*(x2-0.5)<0.2 && x2>0.5)
				classification = 2;
			else if((x1+0.5)*(x1+0.5) + (x2-0.5)*(x2-0.5)<0.2 && x2<0.5)
				classification = 1;
			else
				classification = 3;

			if(i < 4000)
				trainingDataset.write(x1 +","+x2+","+classification+"\n");
			else
				testingDataset.write(x1 +","+x2+","+classification+"\n");
			
		}
		trainingDataset.close();
		testingDataset.close();
	}

}
