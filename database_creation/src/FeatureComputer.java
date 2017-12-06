import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.imgproc.Imgproc;

public class FeatureComputer{

    private class GaborFilter{

        float prf;
        float fo;
        float fb;
        float ab;

        public GaborFilter(float prf,float fo,float fb,float ab){
            this.prf = prf;
            this.fo = fo;
            this.fb = fb;
            this.ab = ab;
        }

        public Mat apply(Mat image){
            Mat padded = null;
            int r = image.rows(), c = image.cols();
            int opt_r = Core.getOptimalDFTSize(r);
            int opt_c = Core.getOptimalDFTSize(c);
            Core.copyMakeBorder(
                    image,
                    padded,
                    0,
                    opt_r-r,
                    0,
                    opt_c-c,
                    Core.BORDER_DEFAULT,
                    Scalar.all(0));
            throw new Error("Non implemented");
        }
    }

    public FeatureComputer(float prfs,float fos,float fbs,float abs){

    }

    public static void main(String[] args){
        return;
    }
}
