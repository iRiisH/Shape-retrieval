import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;


public class ReadViews
{
	static float[][] read_views(int n_views)
	{
		String filepath = "../data/views/"+n_views+".txt";
		FileReader input;
		float[][] res = new float[n_views][3];
		try {
			input = new FileReader(filepath);
			BufferedReader bufRead = new BufferedReader(input);
			String myLine = null;
			
			int cnt = 0;
			try {
				while ( (myLine = bufRead.readLine()) != null)
				{
				    String[] array = myLine.split(" ");
				    int n = array.length;
				    assert (n ==3);
				    for (int i = 0; i < 3; i++)
				        res[cnt][i] = Float.parseFloat(array[i]);
				    cnt++;
				}
			} catch (NumberFormatException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
			try {
				input.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return res;
	}
	static float[][] cart2cam(float[][] cart)
	{
		int n = cart.length;
		float[][] res = new float[n][3];
		for (int i = 0 ; i < n ; i++)
		{
			float x = cart[i][0], y = cart[i][1], z = cart[i][2];
			double theta = Math.asin(x);
			if (z < 0.)
				theta += Math.PI;
			double phi = Math.asin(y/Math.cos(theta));
			if (z/(Math.cos(theta) * Math.cos(phi)) < 0.)
				phi = -phi;
			res[i][0] = (float)phi;
			res[i][1] = (float)theta;
			res[i][2] = 0.f;
		}
		return res;		
	}
	public static void main (String[] args)
	{
		float[][] res = cart2cam(read_views(22));
		int n = res.length;
		for (int i = 0 ; i < n ; i++)
			System.out.println(""+res[i][0]+", "+res[i][1]+", "+res[i][2]);
		
	}
}
