import org.opencv.core.*;
import java.util.*;

public class RetrievalSystem{

    // number of views per mesh
    int num_view;

    // when sampling subimages to compute local features
    // number of rows
    int num_row_sample;
    // number of cols
    int num_col_sample;
    // number of total samples
    int num_sample;

    // number of Gabor filters
    int num_kernels;

    // number of tiles as indicated in article
    int n_tile;

    // feature size as indicated in the article
    float feature_size;

    // fraction of features used to train Kmeans
    float kmeans_train_size;

    // dimension of a feature vector (that used for Kmeans)
    int num_features;

    // size of vocabulary (dimension of histograms)
    int size_vocabulary;

    // prospective trained histograms for each model for each view
    float[][][] histograms;

    // prospective trained Kmeans centroids to discretize feature vectors
    float[][] centroids;

    // feature computer class to compute feature vectors from views
    FeatureComputer fc;

    public RetrievalSystem(
            int num_view_per_img,
            int num_row_sample_per_view,
            int num_col_sample_per_view,
            int num_kernels,
            int n_tile,
            float feature_size,
            float kmeans_train_size,
            int size_vocabulary){
        this.num_view = num_view_per_img;
        this.num_row_sample = num_row_sample_per_view;
        this.num_col_sample = num_col_sample_per_view;
        this.num_sample = this.num_row_sample * this.num_col_sample;
        this.num_kernels = num_kernels;
        this.n_tile = n_tile;
        this.feature_size = feature_size;
        this.kmeans_train_size = kmeans_train_size;
        this.num_features = this.num_kernels * this.n_tile * this.n_tile;
        this.size_vocabulary = size_vocabulary;
        float[][] params = this.getInitalKernelParams();
        this.fc = new FeatureComputer(
                params[0],params[1],params[2],params[3],
                this.num_row_sample,this.num_col_sample,this.feature_size,this.n_tile);
    }

    /*
        receive a SurfaceMesh and compute its views
    */
    private Mat[] render(SurfaceMesh mesh){
        throw new Error("non implemented");
    }


    /*
        get parameters for Gabor filters
    */
    private float[][] getInitalKernelParams(){
        throw new Error("non implemented");
    }

    /*
        check whether a feature vector is blank
    */
    private boolean isZeros(float[] checked){
        for(int i=0;i<checked.length;i++){
            if(checked[i]>0.f)return false;
        }
        return true;
    }

    /*
        sample (v.) features from a set of features
    */
    private float[][] sampleFeatures(float[][][][] local_features){
        int num_mesh = local_features.length;
        int total_local_features = num_mesh * this.num_view * this.num_sample;
        int train_size = (int)(total_local_features * this.kmeans_train_size);
        float[][] sampled_features = new float[train_size][this.num_features];
        ArrayList<Integer> ids = new ArrayList<Integer>(total_local_features);
        for(int i=0;i<total_local_features;i++){ids.add(i);}
        Collections.shuffle(ids);
        for(int i=0,pos=0;i<train_size;i++){
            if(pos>=ids.size()) throw new Error("Too much zero features!");
            while(true){
                int cur = ids.get(pos);
                int a = cur/this.num_view/this.num_sample;
                int b = cur%this.num_view/this.num_sample;
                int c = cur%this.num_sample;
                if(isZeros(local_features[a][b][c]))pos++;
                else{
                    sampled_features[i] = local_features[a][b][c];
                    break;
                }
            }
            pos++;
        }
        return sampled_features;
    }

    /*
        transform a list of features to a list of histograms
    */
    private float[][] getHistograms(float[][][] features){
        float[][] histograms = new float[features.length][this.size_vocabulary];
        for(int i=0;i<features.length;i++){
            for(int j=0;j<this.num_sample;j++){
                if(this.isZeros(features[i][j]))continue;
                int mnid = 0;
                float mndist = KMeans.norm2(this.centroids[0],features[i][j]);
                for(int k=1;k<this.size_vocabulary;k++){
                    float dist = KMeans.norm2(this.centroids[k],features[i][j]);
                    if(dist < mndist){
                        mndist = dist;
                        mnid = k;
                    }
                }
                histograms[i][mnid]++;
            }
        }
        return histograms;
    }

    /*
        transform histograms to TFIDF weights
    */
    private void TFIDF(float[][][] histograms){
        float num_documents = histograms.length * this.num_view;
        float[] freq_documents = new float[this.size_vocabulary];
        for(int i=0;i<histograms.length;i++){
            for(int j=0;j<this.num_view;j++){
                for(int k=0;k<this.size_vocabulary;k++){
                    if(histograms[i][j][k]>0.f){
                        freq_documents[k]++;
                    }
                }
            }
        }
        for(int i=0;i<histograms.length;i++){
            for(int j=0;j<this.num_view;j++){
                float num_words = 0.f;
                for(int k=0;k<this.size_vocabulary;k++){
                    num_words += histograms[i][j][k];
                }
                for(int k=0;k<this.size_vocabulary;k++){
                    histograms[i][j][k] *= Math.log(num_documents/freq_documents[k])/num_words;
                }
            }
        }
    }

    /*
        fit function of retrieval system
    */
    public void fit(SurfaceMesh[] meshes){

        float[][][][] local_features;
        local_features = new float[meshes.length][this.num_view][this.num_sample][this.num_features];

        for(int mesh_id=0;mesh_id<meshes.length;mesh_id++){
            SurfaceMesh mesh = meshes[mesh_id];
            Mat[] imgs = this.render(mesh);
            local_features[mesh_id] = fc.computeFeature(imgs);
        }

        float[][] sampled_features = this.sampleFeatures(local_features);
        KMeans kmeans = new KMeans(this.size_vocabulary);
        kmeans.fit(sampled_features);
        System.out.println("KMeans get centroids:");
        kmeans.printCentroids();
        this.centroids = kmeans.centroids;

        this.histograms = new float[meshes.length][this.num_view][this.size_vocabulary];
        for(int i=0;i<meshes.length;i++){
            this.histograms[i] = this.getHistograms(local_features[i]);
        }
        this.TFIDF(this.histograms);
    }

    /*
        Handy functions to compute cosine similarity between a and b vectors
    */
    public float cosineSimilarity(float[] a,float[] b){
        float norm_a = KMeans.norm2(a);
        float norm_b = KMeans.norm2(b);
        float inner_product = 0.f;
        for(int i=0;i<a.length;i++){
            inner_product += a[i] * b[i];
        }
        return inner_product / norm_a / norm_b;
    }

    /*
        compute similarities for each queries for each 3D meshes
    */
    private float[][] computeSimilarities(float[][] histos){
        int num_meshes = this.histograms.length;
        float[][] sims = new float[histos.length][num_meshes];
        for(int i=0;i<histos.length;i++){
            for(int j=0;j<num_meshes;j++){
                float mxsim = this.cosineSimilarity(this.histograms[j][0],histos[i]);
                for(int k=1;k<this.num_view;k++){
                    mxsim = Math.max(mxsim,this.cosineSimilarity(this.histograms[j][k],histos[i]));
                }
                sims[i][j] = mxsim;
            }
        }
        return sims;
    }

    /*
        argsort
    */
    private int[] argsort(float[] arr){
        throw new Error("Not implemented");
    }

    /*
        give predictions according to similarities
    */
    private int[][] similarities2Predictions(float[][] sims,int num_predictions){
        int[][] predictions = new int[sims.length][num_predictions];
        for(int i=0;i<sims.length;i++){
            int[] ranks = this.argsort(sims[i]);
            for(int j=0;j<num_predictions;j++){
                predictions[i][j] = ranks[j];
            }
        }
        return predictions;
    }

    /*
        predict function of retrieval system
    */
    public int[][] predict(Mat[] sketches,int num_predictions){
        float[][][] features = this.fc.computeFeature(sketches);
        float[][] sketches_histo = this.getHistograms(features);
        float[][] similarities = this.computeSimilarities(sketches_histo);
        int[][] predictions = this.similarities2Predictions(similarities,num_predictions);
        return predictions;
    }
}
