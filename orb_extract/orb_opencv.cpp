#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv){
    if(argc != 3){
        cout << argc << "Usage: img1 img2" << endl;
        return 1;
    }

    Mat img1 = imread(argv[1], IMREAD_COLOR);
    Mat img2 = imread(argv[2], IMREAD_COLOR);
    assert(img1.empty() || img2.empty());

    vector<KeyPoint> kp1, kp2;
    Mat dpct1, dpct2;
    
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    detector->detect(img1,kp1);
    detector->detect(img2,kp2);

    descriptor->compute(img1, kp1, dpct1);
    descriptor->compute(img2, kp2, dpct2);

    auto duration = chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now()-t1);

    cout << "extract ORB cost = " << duration.count() << " seconds. " << endl;

    Mat outimg1;
    drawKeypoints(img1, kp1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features", outimg1);

    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(dpct1, dpct2, matches);
    duration = chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now()-t1);

    cout << "match ORB cost = " << duration.count() << " seconds. " << endl;

    auto min_max = minmax_element(matches.begin(),matches.end(),
                                 [](const DMatch &m1, const DMatch &m2) {return m1.distance < m2.distance; });

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    cout << "Max dist:" << max_dist << endl;
    cout << "Min dist:" << min_dist << endl;

    vector<DMatch> good_matches;
    for(int i = 0; i < dpct1.rows; i++){
        if(matches[i].distance <= max(2*min_dist, 30.0))  {
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_match, img_goodmatch;
    drawMatches(img1, kp1, img2, kp2, matches, img_match);
    drawMatches(img1, kp1, img2, kp2, good_matches, img_goodmatch);

    imshow("all matches",img_match);
    imshow("good mathces", img_goodmatch);

    waitKey(0);

    return 0;
}