import java.util.*;

public class KMeans{

    public Random random = new Random();
    public int num_cluster;
    public int num_iter = 100;
    public int dim,n;
    public float[][] centroids;
    public int[] labels;
    public int[] counts;
    public float[][] data;

    public KMeans(int num_cluster){
        this.num_cluster = num_cluster;
    }

    public void setNumIter(int num_iter){
        this.num_iter = num_iter;
    }

    public void initialize(){
        ArrayList<Integer> to_shuffle = new ArrayList<Integer>(this.n);
        for(int i=0;i<this.n;i++){
            to_shuffle.add(i);
        }
        Collections.shuffle(to_shuffle,this.random);
        this.centroids = new float[this.num_cluster][this.dim];
        for(int i=0;i<this.num_cluster;i++){
            this.centroids[i] = this.data[to_shuffle.get(i)];
        }
        // this.printCentroids();
    }

    public int emptyIndex(){
        for(int i=0;i<this.num_cluster;i++){
            if(this.counts[i]==0){
                return i;
            }
        }
        return -1;
    }

    public boolean guaranteeNonEmpty(){
        int empty_id = this.emptyIndex();
        if(empty_id<0) return true;
        int picked = this.random.nextInt(this.n);
        this.centroids[empty_id] = this.data[picked];
        return false;
    }

    public static float norm2(float[] a,float[] b){
        float r = 0.f;
        for(int i=0;i<a.length;i++){
            r += (float)Math.pow(a[i]-b[i],2);
        }
        return (float)Math.sqrt(r);
    }

    public void distribute(){
        this.counts = new int[this.num_cluster];
        this.labels = new int[this.n];
        for(int i=0;i<this.n;i++){
            int mnid = 0;
            float mndist = norm2(this.data[i],centroids[0]);
            for(int j=1;j<this.num_cluster;j++){
                float dist = norm2(this.data[i],this.centroids[j]);
                if(dist < mndist){
                    mnid = j;
                    mndist = dist;
                }
            }
            this.labels[i] = mnid;
            this.counts[mnid]++;
        }
    }

    public void getCentroids(){
        this.centroids = new float[this.num_cluster][this.dim];
        for(int i=0;i<this.n;i++){
            for(int j=0;j<dim;j++){
                this.centroids[this.labels[i]][j] += this.data[i][j]/(float)this.counts[this.labels[i]];
            }
        }
    }

    public void fit(float[][] data){
        this.data = data;
        this.n = data.length;
        this.dim = data[0].length;
        this.initialize();
        for(int i=0;i<this.num_iter;i++){
            while(true){
                this.distribute();
                if(this.guaranteeNonEmpty())break;
                System.out.println("Empty case");
            }
            this.getCentroids();
            // this.printCounts();
            // this.printCentroids();
            // this.printLabels();
        }
    }

    public float divergence(){
        float div = 0.f;
        for(int i=0;i<this.n;i++){
            div += this.norm2(this.centroids[this.labels[i]],this.data[i]);
        }
        return div;
    }

    public void printCentroids(){
        for(int i=0;i<this.num_cluster;i++){
            for(int j=0;j<this.centroids[0].length;j++){
                System.out.print(this.centroids[i][j]);
                System.out.print(' ');
            }
            System.out.println();
        }
    }
    public void printCounts(){
        for(int i=0;i<this.counts.length;i++){
            System.out.print(this.counts[i]);
        }
        System.out.println();
    }
    public void printLabels(){
        for(int i=0;i<this.labels.length;i++){
            System.out.print(this.labels[i]);
        }
        System.out.println();
    }

    public static void main(String[] args){
        float[][] data = {{.1f,.1f},{-.1f,.1f},{.1f,-.1f},{-.1f,-.1f},
                        {1.1f,.1f},{.9f,.1f},{1.1f,-.1f},{.9f,-.1f},
                        {1.1f,1.1f},{.9f,.9f},{1.1f,.9f},{.9f,1.1f},
                        {.1f,1.1f},{-.1f,1.1f},{.1f,.9f},{-.1f,.9f}};
        KMeans kmeans = new KMeans(5);
        kmeans.fit(data);
        kmeans.printLabels();
        System.out.println(kmeans.divergence());
    }
}
