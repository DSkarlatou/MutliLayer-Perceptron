public class Layer {
    private static final int min = -1;
    private static final int max = 1;

    float[] weight, errors;
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
