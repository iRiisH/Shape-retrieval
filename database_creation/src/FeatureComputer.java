import processing.core.*;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.imgcodecs.*;
import java.awt.image.*;
// import java.awt.image.BufferedImage;
import java.util.*;
// import java.awt.*;
import java.awt.Image;
import java.awt.FlowLayout;
import javax.swing.*;

public class FeatureComputer{

    public GaborFilter[] gbfs;
    public int row_sample, col_sample;
    public float feature_size;
    public int n_tile;

    private class GaborFilter{

        float prf;
        float fo;
        float fb;
        float ab;
        float cfo,sfo;


        public GaborFilter(float prf,float fo,float fb,float ab){
            this.prf = prf;
            this.fo = fo;
            this.cfo = (float)Math.cos(fo);
            this.sfo = (float)Math.sin(fo);
            this.fb = fb;
            this.ab = ab;
        }

        private Mat DFT(Mat image, int inverse){
            Mat padded = new Mat();
            Mat complexImage = new Mat();
            Mat mag = new Mat();
            ArrayList<Mat> newPlanes = new ArrayList<Mat>();
            ArrayList<Mat> planes = new ArrayList<Mat>();
            int addPixelRows = Core.getOptimalDFTSize(image.rows());
            int addPixelCols = Core.getOptimalDFTSize(image.cols());
            Core.copyMakeBorder(
                    image,
                    padded,
                    0,
                    addPixelRows - image.rows(),
                    0,
                    addPixelCols - image.cols(),
                    Core.BORDER_CONSTANT,
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

        private float Gabor_func(float u,float v){
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
                    float val = this.Gabor_func((r/1.f/R),(c/1.f/C));
                    float[] vals = {val,val};
                    kernel.put(r,c,vals);
                    double[] pixel = kernel.get(r,c);
                    // for(int i=0;i<pixel.length;i++){
                    //     System.out.print(pixel[i]);
                    //     System.out.print(' ');
                    // }
                    // System.out.println("ed");
                }
            }
            return kernel;
        }

        public Mat apply(Mat image){
            Range range = new Range(0,image.rows());
            Mat dft_I = new Mat(image,range);
            Mat idft_I = new Mat(image,range);
            Mat kernel = new Mat(image,range);
            dft_I = this.DFT(image,-1);
            kernel = this.getKernel(dft_I);
            ArrayList<Mat> ks = new ArrayList<Mat>();
            Core.split(kernel,ks);
            displayMat(ks.get(0));
            idft_I = this.DFT(dft_I.mul(kernel),Core.DFT_INVERSE);
            // idft_I = this.DFT(dft_I,Core.DFT_INVERSE);
            // printMat(kernel);
            // displayMat(kernel);
            // printMat(kernel);
            // Core.dft( idft_I, Core.DFT_INVERSE, 0);
            return idft_I;
        }
    }

    public FeatureComputer(float[] prfs,float[] fos,float[] fbs,float[] abs,
            int row_sample, int col_sample, float feature_size, int n_tile){
        int l = prfs.length;
        this.gbfs = new GaborFilter[l];
        for(int i=0;i<l;i++){
            this.gbfs[i] = new GaborFilter(prfs[i],fos[i],fbs[i],abs[i]);
        }
        this.row_sample = row_sample;
        this.col_sample = col_sample;
        this.feature_size = feature_size;
        this.n_tile = n_tile;
    }

    public Mat[][] computeResponseImage(Mat[] ms){
        Mat[][] res = new Mat[ms.length][gbfs.length];
        for(int k=0;k<ms.length;k++){
            for(int i=0;i<gbfs.length;i++){
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
        int R = rimg[0].rows(), C = rimg[0].cols();
        if(cf[0]<0||cf[1]<0||cf[2]>=R||cf[3]>=C)return res;
        int rs = rimg.length;
        float wh = (cf[2]-cf[0])/this.n_tile;
        for(int i=0;i<rs;i++){
            for(int j=0;j<this.n_tile;j++){
                for(int k=0;k<this.n_tile;k++){
                    float u = cf[0]+j*wh, l = cf[1]+k*wh;
                    float d = cf[0]+(j+1)*wh, r = cf[1]+(k+1)*wh;
                    float val=0.f;
                    for(float c1=u;c1<d;c1++){
                        for(float c2=l;c2<r;c2++){
                            int c1bed = (int)c1;
                            int c2bed = (int)c2;
                            float p1 = c1 - c1bed;
                            float p2 = c2 - c2bed;
                            val += (float)(p1*p2*rimg[i].get(c1bed,c2bed)[0]);
                            val += (float)((1-p1)*p2*rimg[i].get(c1bed+1,c2bed)[0]);
                            val += (float)(p1*(1-p2)*rimg[i].get(c1bed,c2bed+1)[0]);
                            val += (float)((1-p1)*(1-p2)*rimg[i].get(c1bed+1,c2bed+1)[0]);
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
        for(int view_id = 0;view_id < rimg.length;view_id++){
            for(int frame_id = 0;frame_id < frames.length;frame_id++){
                float[] cf = frames[frame_id];
                features[view_id][frame_id] = this.computeFrameFeatures(cf,rimg[view_id]);
            }
        }
        return features;
    }

    public float[][][] computeFeature(Mat[] ms){
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

    public static void main(String[] args){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Viewer viewer = new Viewer();
        viewer.setSize(400, 400);
		PApplet.main(new String[] { "Viewer" });
        PImage img = viewer.get();
        // PImage img_ = img.get();
        BufferedImage bimg = (BufferedImage)img.getImage();
        byte[] pixels = ((DataBufferByte) bimg.getRaster().getDataBuffer())
            .getData();

        // Create a Matrix the same size of image
        Mat image = new Mat(bimg.getHeight(), bimg.getWidth(), CvType.CV_8UC3);
        // Fill Matrix with image values
        image.put(0, 0, pixels);
        displayMat(image);
        // Mat[] m = {Imgcodecs.imread("two-nodes.png",Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE)};
        // float[] prfs = {0.1f};
        // float[] fos = {(float)Math.PI/4.f};
        // float[] fbs = {5.f};
        // float[] abs = {10.f};
        // FeatureComputer fc = new FeatureComputer(prfs,fos,fbs,abs,32,32,0.075f,4);
        // Mat[][] ms = fc.computeResponseImage(m);
        // displayMat(ms[0][0]);
        return;
    }
}
