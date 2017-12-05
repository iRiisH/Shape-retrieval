// import java.util.Properties;
// import java.util.Enumeration;
//
// public class FeatureComputer {
//   public static void main(String args[]) {
//     if( args.length == 0 ) {
//       Properties p = System.getProperties();
//       Enumeration keys = p.keys();
//       while (keys.hasMoreElements()) {
//         String key = (String)keys.nextElement();
//         String value = (String)p.get(key);
//         System.out.println(key + " : " + value);
//       }
//     }
//     else {
//       for (String key: args) {
//         System.out.println(System.getProperty( key ));
//       }
//     }
//   }
// }


import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class FeatureComputer
{
   public static void main( String[] args )
   {
       // System.out.println(Core.NATIVE_LIBRARY_NAME);
      // System.loadLibrary( "opencv-331" );
      Mat mat = Mat.eye( 3, 3, CvType.CV_8UC1 );
      System.out.println( "mat = " + mat.dump() );
   }
}

// //
// // public class FeatureComputer{
// //
// //     private class GaborFilter{
// //
// //         float prf;
// //         float fo;
// //         float fb;
// //         float ab;
// //
// //         public GaborFilter(float prf,float fo,float fb,float ab){
// //             this.prf = prf;
// //             this.fo = fo;
// //             this.fb = fb;
// //             this.ab = ab;
// //         }
// //
// //         public Mat apply(Mat m){
// //             Mat padded;
// //             int r = image.rows(), c = image.cols();
// //             int opt_r = Core.getOptimalDFTSize(r);
// //             int opt_c = Core.getOptimalDFTSize(c);
// //             Core.copyMakeBorder(
// //                     image,
// //                     padded,
// //                     0,
// //                     opt_r-r,
// //                     0,
// //                     opt_c-c,
// //                     Imgproc.BORDER_CONSTANT,
// //                     Scalar.all(0));
// //         }
// //     }
// //
// //     public FeatureComputer(float prfs,float fos,float fbs,float abs){
// //
// //     }
// // }
