import processing.core.*;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;

import java.awt.image.*;
import java.util.*;
import java.awt.Image;
import java.awt.FlowLayout;
import javax.swing.*;

public class FeatureComputer{

    public GaborFilter[] gbfs;
    public int row_sample, col_sample;
    public float feature_size;
    public int n_tile;

    private class GaborFilter{

    	double sigma, theta, Lambda, psi, gamma;

   
    	
        public GaborFilter(double sigma, double theta, double Lambda, double psi, double gamma){
            this.sigma = sigma;
            this.theta = theta;
            this.Lambda = Lambda;
            this.psi = psi;
            this.gamma = gamma;
        }

        private Mat DFT(Mat image, int inverse)
        /* 
         * discrete Fourier transform of image
         */
        {
            Mat padded = new Mat();
            Mat complexImage = new Mat();
            Mat mag = new Mat();
            ArrayList<Mat> newPlanes = new ArrayList<Mat>();
            ArrayList<Mat> planes = new ArrayList<Mat>();
            int addPixelRows = Core.getOptimalDFTSize(image.rows());
            int addPixelCols = Core.getOptimalDFTSize(image.cols());
            Imgproc.copyMakeBorder(
                    image,
                    padded,
                    0,
                    addPixelRows - image.rows(),
                    0,
                    addPixelCols - image.cols(),
                    Imgproc.BORDER_CONSTANT,
                    Scalar.all(0));
            padded.convertTo(padded, CvType.CV_32F);
            planes.add(padded);
            planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
            Core.merge(planes, complexImage);
            complexImage.convertTo(complexImage,CvType.CV_32F);
            if(inverse >= 0){
                Core.idft(image, image);
                Mat restoredImage = new Mat();
                Core.split(image, planes);
                Core.normalize(planes.get(0), restoredImage, 0, 255, Core.NORM_MINMAX);
                return restoredImage;
            }
            else{
                Core.dft(complexImage, complexImage);
                return complexImage;
            }

        }
        
        private double[][] gabor_field()
        {
        	double sigma_x = this.sigma;
        	double sigma_y = this.sigma / this.gamma;


        	int nstds = 5; //Number of standard deviation sigma
        	double xmax = Math.max(Math.abs(nstds * sigma_x * Math.cos(this.theta)), Math.abs(nstds * sigma_y * Math.sin(this.theta)));
        	int xmax_i = (int)Math.ceil(Math.max(1, xmax));
        	double ymax = Math.max(Math.abs(nstds * sigma_x * Math.sin(this.theta)), Math.abs(nstds * sigma_y * Math.cos(this.theta)));
        	int ymax_i = (int)Math.ceil(Math.max(1, ymax));
        	int xmin_i = -xmax_i;
        	int ymin_i = -ymax_i;
        	double[][] result = new double[xmax_i+1-xmin_i][ymax_i+1-ymin_i];
        	for (int i = xmin_i ; i < xmax_i+1 ; i++)
        	{
        		for (int j = ymin_i ; j < ymax_i+1 ; j++)
        		{
        			double y = (double)j;
        			double x = (double)i;
        			// Rotation 
        		    double x_theta = x * Math.cos(this.theta) + y * Math.sin(this.theta);
        		    double y_theta = -x * Math.sin(this.theta) + y * Math.cos(this.theta);
        		    double gb = Math.exp(-.5 * (Math.pow(x_theta,  2.) / Math.pow(sigma_x, 2.) + Math.pow(y_theta, 2.) /
        		    		Math.pow(sigma_y,2.))) * Math.cos(2 * Math.PI / this.Lambda * x_theta + this.psi);
        		    result[i+xmax_i][j+ymax_i] = gb;

        		}
        	}
        	
        	return result;
        }
        
        private double secured_val(Mat img, int i, int j)
        // returns the value at pixel (i, j), 0. if outside image bounds.
        // Assumes the image is grayscale
        {
        	int h = img.height(), w = img.width();
        	if (i < 0 || i >= h || j < 0 || j >= w)
        		return 0.;
        	double[] pix = img.get(i, j);
        	return pix[0]; // if the image is indeed grayscale all values should be equal       	
        }
        
        
        private double[][] convolve(Mat img, double[][] gabor)
        /*
         * convolves img with gabor filter
         */
        {
        	int h = img.height(), w = img.width();
        	double[][] arr = new double[h][w];
        	for (int i = 0 ; i < h ; i++)
        	{
        		for (int j = 0 ; j < w ; j++)
        		{
        			arr[i][j] = img.get(i, j)[0];
        		}
        	}
        	
        	System.out.println(h);
        	System.out.println(w);
        	System.out.println(gabor.length);
        	System.out.println(gabor[0].length);
        	
        	return Convolution.convolution2D(arr, h, w, gabor, gabor.length, gabor[0].length);
        }
        
        /*private float Gabor_func(float u,float v)
        {
            float _u = cfo * u - sfo * v;
            float _v = sfo * u + cfo * v;
            float res = 256*(float)Math.exp(-2.f*Math.pow(Math.PI,2)
                    *(Math.pow((_u-this.prf)*this.fb,2)+Math.pow(_v*this.ab,2)));
            return res;
        }

        private Mat getKernel(Mat image){
            Mat kernel = image.clone();
            int R = image.rows(), C = image.cols();
            for(int r=0;r<R;r++){
                for(int c=0;c<C;c++){
                    // System.out.println(r);
                    // System.out.println(R);
                    // System.out.println(c);
                    // System.out.println(C);
                    float val = this.Gabor_func((r/1.f/R*2.f-1.f),(c/1.f/C*2.f-1.f));
                    float[] vals = {val,val};
                    kernel.put(r,c,vals);
                    double[] pixel = kernel.get(r,c);
                    // for(int i=0;i<pixel.length;i++){
                    //     System.out.print(pixel[i]);
                    //     System.out.print(' ');
                    // }double[]
                    // System.out.println("ed");
                }
            }
            return kernel;
        }*/

        public Mat apply(Mat image)
        {
           	double[][] kernel = this.gabor_field();
        	double[][] res = this.convolve(image, kernel);
        	int h = res.length, w = res[0].length;
        	Mat res_ = new Mat(h, w, CvType.CV_64FC1);
        	for (int i = 0 ; i  < h ; i++)
        	{
        		for (int j = 0 ; j < w ; j++)
        		{
        			res_.put(i, j, res[i][j]);
        		}
        	}
        	return res_;
        	
            /*Range range = new Range(0,image.rows());
            Mat dft_I = new Mat(image,range);
            Mat idft_I = new Mat(image,range);
            Mat kernel = new Mat(image,range);
            dft_I = this.DFT(image,-1);
            kernel = this.getKernel(dft_I);
            ArrayList<Mat> ks = new ArrayList<Mat>();
            Core.split(kernel,ks);
            // displayMat(ks.get(0));
            idft_I = this.DFT(dft_I.mul(kernel),Core.DFT_INVERSE);
            // displayMat(idft_I);
            return idft_I;*/
        }
    }

    public FeatureComputer(float[] theta, int row_sample, int col_sample, float feature_size, int n_tile){
    	double sigma = 2.;
    	double lambda = 0.43;
    	double psi = 0.;
    	double gamma = 1.;
        int l = theta.length;
        this.gbfs = new GaborFilter[l];
        for(int i=0;i<l;i++){
            this.gbfs[i] = new GaborFilter(sigma, (double)theta[i], lambda, psi, gamma);
        }
        this.row_sample = row_sample;
        this.col_sample = col_sample;
        this.feature_size = feature_size;
        this.n_tile = n_tile;
    }

    public Mat[][] computeResponseImage(Mat[] ms){
        Mat[][] res = new Mat[ms.length][gbfs.length];
        System.out.println("Computing response images...");
        for(int k=0;k<ms.length;k++){
            for(int i=0;i<gbfs.length;i++){
                System.out.println("View id "+String.valueOf(k)+" Filter id "+String.valueOf(i));
                res[k][i] = gbfs[i].apply(ms[k]);
            }
        }
        return res;
    }

    private float[][] computeCoords(int row,int col){
        float[][] coords = new float[this.row_sample*this.col_sample][4];
        float dist_row = row * 1.f / this.row_sample;
        float dist_col = col * 1.f / this.col_sample;
        float len = (float)Math.sqrt(row * col * this.feature_size);
        // len = Math.min(len,Math.min(dist_row,dist_col));
        for(int r=0;r<this.row_sample;r++){
            for(int c=0;c<this.col_sample;c++){
                coords[r*this.col_sample+c][0] = (dist_row * (.5f + r) - len*.5f);
                coords[r*this.col_sample+c][1] = (dist_col * (.5f + c) - len*.5f);
                coords[r*this.col_sample+c][2] = (dist_row * (.5f + r) + len*.5f);
                coords[r*this.col_sample+c][3] = (dist_col * (.5f + c) + len*.5f);
            }
        }
        return coords;
    }

    private Rect getRect(int a,int b,int c,int d){
        return new Rect(new Point(a,b),new Point(c,d));
    }

    private float[] computeFrameFeatures(float[] cf,Mat[] rimg){
        float[] res = new float[this.gbfs.length*this.n_tile*this.n_tile];
        int R = rimg[0].cols(), C = rimg[0].rows();
        if(cf[0]<0||cf[1]<0||cf[2]+1>=R||cf[3]+1>=C)return res;
        int rs = rimg.length;
        float wh = (cf[2]-cf[0])/this.n_tile;
        float p1 = cf[0] - (int)cf[0], p2 = cf[1] - (int)cf[1];
        int[] icf = {(int)cf[0],(int)cf[1],(int)cf[2],(int)cf[3]};
        Rect[] rects = {
                this.getRect(icf[0],icf[1],icf[2],icf[3]),
                this.getRect(icf[0]+1,icf[1],icf[2]+1,icf[3]),
                this.getRect(icf[0],icf[1]+1,icf[2],icf[3]+1),
                this.getRect(icf[0]+1,icf[1]+1,icf[2]+1,icf[3]+1)};
        for(int i=0;i<rs;i++){
        	/*int[][] non_zero_coord = non_zero_pix.get(i);
        	int x_min = rimg[i].width(), x_max = 0, y_min = rimg[i].height(), y_max = 0;
        	for (int k = 0 ; k < rimg[i].width ; k++)
        	{
        		
        	}*/
            Mat ul = new Mat(rimg[i],rects[0]);
            Mat dl = new Mat(rimg[i],rects[1]);
            Mat ur = new Mat(rimg[i],rects[2]);
            Mat dr = new Mat(rimg[i],rects[3]);
            Mat piece = new Mat();
            Core.multiply(ul,new Scalar(p1*p2),ul);
            Core.multiply(dl,new Scalar((1-p1)*p2),ul);
            Core.multiply(ur,new Scalar(p1*(1-p2)),ul);
            Core.multiply(dr,new Scalar((1-p1)*(1-p2)),ul);
            Core.add(ul,dl,piece);
            Core.add(piece,ur,piece);
            Core.add(piece,dr,piece);
            if(Core.minMaxLoc(piece).maxVal<1.0)continue;
            for(int j=0;j<this.n_tile;j++){
                for(int k=0;k<this.n_tile;k++){
                    int u = icf[0]+(int)(j*wh), l = icf[1]+(int)(k*wh);
                    int d = Math.min(icf[0]+(int)((j+1)*wh),icf[2]+1), r = Math.min(icf[1]+(int)((k+1)*wh),icf[3]+1);
                    float val=0.f;
                    for(int c1=u;c1<d;c1++){
                        for(int c2=l;c2<r;c2++){
                            val += (float)piece.get(c1-u,c2-l)[0];
                        }
                    }
                    res[i*this.n_tile*this.n_tile+j*this.n_tile+k] = val;
                }
            }
        }
        return res;
    }

    private float[][][] computeLocalFeatures(Mat[][] rimg,float[][] frames){
        /*
            Output contains blank features
        */

        float[][][] features;
        features = new float[rimg.length][frames.length][this.gbfs.length*this.n_tile*this.n_tile];
        System.out.println("Computing local features...");
        for(int view_id = 0;view_id < rimg.length;view_id++){
            for(int frame_id = 0;frame_id < frames.length;frame_id++){
                System.out.println("\rView id "+String.valueOf(view_id)+" frame id "+String.valueOf(frame_id));
                float[] cf = frames[frame_id];
                features[view_id][frame_id] = this.computeFrameFeatures(cf,rimg[view_id]);
            }
        }
        return features;
    }

    public float[][][] computeFeature(Mat[] ms)
    {
    	/*int n_imgs = ms.length;
    	LinkedList<int[][]> non_zero_pix = new LinkedList<int[][]>();
    	for (int i = 0 ; i < ms.length; i++)
    	{
    		int w = ms[i].width(), h = ms[i].height();
    		LinkedList<int[]> ll = new LinkedList<int[]>();
    		int[][] non_zero_coord;
    		for (int k = 0 ; k < w ; k++)
    		{
    			for (int l = 0 ; l < h ; l++)
    			{
    				double[] pix = ms[i].get(k, l);
    				if (pix[0] > 0.)
    				{
    					int[] coord = {k, l};
    					ll.add(coord);
    				}
    			}
    			int n_non_zero = ll.size();
    			non_zero_coord = ll.toArray(new int[n_non_zero][2]);
    			non_zero_pix.add(non_zero_coord);
    			System.out.println(n_non_zero);
    		}
    	}*/
        Mat[][] rimg = this.computeResponseImage(ms);
        float[][] frames = this.computeCoords(rimg[0][0].rows(),rimg[0][0].cols());
        float[][][] features = computeLocalFeatures(rimg,frames);
        return features;
    }

    public static BufferedImage Mat2BufferedImage(Mat m){
        // source: http://answers.opencv.org/question/10344/opencv-java-load-image-to-gui/
        // Fastest code
        // The output can be assigned either to a BufferedImage or to an Image

        int type = BufferedImage.TYPE_BYTE_GRAY;
        if ( m.channels() > 1 ) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels()*m.cols()*m.rows();
        byte [] b = new byte[bufferSize];
        int mt = m.type();
        m.convertTo(m,16);
        m.get(0,0,b); // get all the pixels
        m.convertTo(m,mt);
        BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    public static void displayImage(Image img2){
        //BufferedImage img=ImageIO.read(new File("/HelloOpenCV/lena.png"));
        ImageIcon icon=new ImageIcon(img2);
        JFrame frame=new JFrame();
        frame.setLayout(new FlowLayout());
        frame.setSize(img2.getWidth(null)+50, img2.getHeight(null)+50);
        JLabel lbl=new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public static void displayMat(Mat m){
        displayImage(Mat2BufferedImage(m));
    }

    public static void printMat(Mat m){
        for(int r=0;r<m.rows();r++){
            for(int c=0;c<m.cols();c++){
                double[] db = m.get(r,c);
                System.out.format("%.0f",db[0]);
                System.out.print(' ');
            }
            System.out.println();
        }
        System.out.println();
        System.exit(0);
    }
}
