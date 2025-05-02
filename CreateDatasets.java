import java.io.FileWriter;
import java.io.IOException;

public class CreateDatasets {
	private static final int min = -1;
	private static final int max = 1;

	public CreateDatasets() throws IOException {
		writeData();
	}

	public static void writeData() throws IOException {
		FileWriter trainingDataset = new FileWriter("training.txt");
		FileWriter testingDataset = new FileWriter("testing.txt");

		for(int i = 0; i < 8000; i++)
		{
			float x1 = generate();
			float x2 = generate();
			int classification;

			boolean b1 = (x1 - 0.5) * (x1 - 0.5) + (x2 - 0.5) * (x2 - 0.5) < 0.2;
			boolean b2 = (x1 + 0.5) * (x1 + 0.5) + (x2 + 0.5) * (x2 + 0.5) < 0.2;
			boolean b3 = (x1 - 0.5) * (x1 - 0.5) + (x2 + 0.5) * (x2 + 0.5) < 0.2;
			boolean b4 = (x1 + 0.5) * (x1 + 0.5) + (x2 - 0.5) * (x2 - 0.5) < 0.2;
			if(b1 && x2>0.5)
				classification = 1;
			else if(b1 && x2<0.5)
				classification = 2;
			else if(b2 && x2>-0.5)
				classification = 1;
			else if(b2 && x2<-0.5)
				classification = 2;
			else if(b3 && x2>-0.5)
				classification = 1;
			else if(b3 && x2<-0.5)
				classification = 2;
			else if(b4 && x2>0.5)
				classification = 2;
			else if(b4 && x2<0.5)
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

	public static float generate()
	{
		return (float) (min + Math.random() * (max - min));
	}

}
