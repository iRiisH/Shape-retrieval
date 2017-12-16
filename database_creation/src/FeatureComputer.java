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

    // Gabor filter
    public GaborFilter[] gbfs;

    // parameters passed from RetrievalSystem
    public int row_sample, col_sample;
    public float feature_size;
    public int n_tile;

    private class GaborFilter{

    	double sigma, theta, Lambda, psi, gamma;
        double[][] kernel;

        public GaborFilter(double sigma, double theta, double Lambda, double psi, double gamma){
            this.sigma = sigma;
            this.theta = theta;
            this.Lambda = Lambda;
            this.psi = psi;
            this.gamma = gamma;
            this.kernel = this.gabor_field();
        }

        // give a gabor kernel in double[][] (Code taken)
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

        // convolve a Mat img with the gabor
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
        	return Convolution.convolution2D(arr, h, w, gabor, gabor.length, gabor[0].length);
        }

        public Mat apply(Mat image)
        {
        	double[][] res = this.convolve(image, this.kernel);
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

    // convolve with Gabor filters
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

    // compute coordinates of frames for a given size (row * col)
    private float[][] computeCoords(int row,int col){
        float[][] coords = new float[this.row_sample*this.col_sample][4];
        float dist_row = row * 1.f / this.row_sample;
        float dist_col = col * 1.f / this.col_sample;
        float len = (float)Math.sqrt(row * col * this.feature_size);
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

    // get a frame with coordinates (up left down right)
    private Rect getRect(int a,int b,int c,int d){
        return new Rect(new Point(a,b),new Point(c,d));
    }

    /*
        compute features for a given frame (cf) and a set of response images (rimg)
    */
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

            // interpolate to get subimage
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

            // if is blank, ignored
            if(Core.minMaxLoc(piece).maxVal<1.0)continue;

            // reduce resolution
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

    /*
        Compute local features from given positions of batches (frames)
    */
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

    /*
        main API to call to compute features from given images (sketches or views)
    */
    public float[][][] computeFeature(Mat[] ms)
    {
        Mat[][] rimg = this.computeResponseImage(ms);
        float[][] frames = this.computeCoords(rimg[0][0].rows(),rimg[0][0].cols());
        float[][][] features = computeLocalFeatures(rimg,frames);
        return features;
    }

    // Debug functions
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

    // Debug functions taken from Internet
    public static BufferedImage Mat2BufferedImage(Mat m){
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
}
