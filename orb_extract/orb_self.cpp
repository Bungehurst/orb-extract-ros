#include "include/orb_self.hpp"

using namespace std;
using namespace cv;

typedef vector<uint32_t> DescType;

void ComputeORB(const Mat &img, vector<KeyPoint> &keypoints, vector<DescType> &descriptors){
    const int half_patch_size = 8;
    const int half_boundary = 16;
    int bad_points = 0;
    
    for(auto &kp:keypoints){
        // remove boarder points 
        if (kp.pt.x < half_boundary || kp.pt.y < half_boundary || 
                kp.pt.x >= img.cols - half_boundary || kp.pt.y >= img.rows - half_boundary){
            bad_points++;
            descriptors.push_back({});
            continue;
        }

        // compute center of mass
        float m10 = 0, m01 = 0;
        for (int dx = -half_patch_size; dx < half_patch_size; ++dx){
            for (int dy = -half_patch_size; dy < half_patch_size; ++dy){
                uchar pixel = img.at<uchar>(kp.pt.y + dy, kp.pt.x + dx);
                m10 += dx * pixel;
                m01 += dy * pixel;
            }
        }

        // compute vector angle, Oreinted FAST
        float m_sqrt = sqrt(pow(m01,2)+pow(m10,2)) + 1e-18; // avoid divide by zero
        float sin_ = m01 / m_sqrt;
        float cos_ = m10 / m_sqrt;

        // compute BRIEF descriptors
        DescType desc(8,0); // 8 dimensions' descriptor
        for (int i = 0; i < 8; i++){
            uint32_t d = 0;
            for (int k = 0; k < 32; k++){
                int idx_pq = i * 32 + k;
                // ORB_pattern is a random array which represents random location on the image
                Point2f p(ORB_pattern[idx_pq * 4], ORB_pattern[idx_pq * 4 + 1]);
                Point2f q(ORB_pattern[idx_pq * 4 + 2], ORB_pattern[idx_pq * 4 + 3]);

                Point2f pp = Point2f(cos_ * p.x - sin_ * p.y, sin_ * p.x + cos_ * p.y) + kp.pt;
                Point2f qq = Point2f(cos_ * q.x - sin_ * q.y, sin_ * q.x + cos_ * q.y) + kp.pt;

                if(img.at<uchar>(pp.y,pp.x) < img.at<uchar>(qq.y, qq.x)) {
                    d |= 1 << k;
                }
            }
            desc[i] = d;
        }
        descriptors.push_back(desc);
    }
    cout << "bad/total: " << bad_points << "/" << keypoints.size() << endl;

} 

// BruteForce Matching
void BfMatch(const vector<DescType> &desc1, const vector<DescType> &desc2, vector<DMatch> &matches){
    const int d_max = 40;

    for (size_t i1 = 0; i1 < desc1.size(); ++i1){
        if (desc1[i1].empty()) continue;
        DMatch maxd{i1,0,256};
        for (size_t i2 = 0; i2 < desc2.size(); ++i2){
            if (desc2[i2].empty()) continue;
            int dist = 0;
            for (int k = 0; k < 8; k++){
                dist += _mm_popcnt_u32(desc1[i1][k] ^ desc2[i2][k]);
            }
            if (dist < d_max && dist < maxd.distance) {
                maxd.distance = dist;
                maxd.trainIdx = i2;
            }
        }
        if (maxd.distance < d_max) {
            matches.push_back(maxd);
        }
    }
}

int main(int argc, char **argv){
    if(argc != 3){
        cout << argc << "Usage: img1 img2" << endl;
        return 1;
    }

    Mat img1 = imread(argv[1], IMREAD_COLOR);
    Mat img2 = imread(argv[2], IMREAD_COLOR);
    assert(img1.empty() || img2.empty());

    // compute Oriented-FAST features and ORB descriptors
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<KeyPoint> kp1,kp2;
    FAST(img1, kp1, 40);
    FAST(img2, kp2, 40);
    vector<DescType> dpct1,dpct2;
    ComputeORB(img1, kp1, dpct1);
    ComputeORB(img2, kp2, dpct2);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

    // find matches 
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    BfMatch(dpct1,dpct2,matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;
    cout << "matches: " << matches.size() << endl;

    Mat img_out;
    drawMatches(img1,kp1,img2,kp2,matches,img_out);
    imshow("Matches",img_out);
    waitKey(0);

    return 0;
}