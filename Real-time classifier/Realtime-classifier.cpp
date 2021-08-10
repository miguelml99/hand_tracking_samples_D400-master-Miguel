
#include <exception>  // base class for standard expceptions (  try throw catch )
#include <iostream>
#include <fstream>
#include <cctype>    // std::tolower converts uppercase letter to lowercase
#include <future>   //Header with facilities that allow asynchronous access to values set by specific providers, possibly in a different thread
#include <sstream>  // Stream class to operate on strings. In C++ stream refers to the stream of characters that are transferred between the program thread and i/o.
#include <algorithm>  // std::max
#include <immintrin.h>  // rdtsc


#include "../include/physmodel.h"    //simulating physically articulated skeletal models and point cloud fitting
#include "../include/handtrack.h"  //articulated hand skeletal model tracking system
#include "../include/dataset.h"     //has the support code for serializing depth/ir data and the ground truth pose information
#include "../include/dcam.h"
#include "../include/misc_image.h"
#include "../third_party/misc.h"   // for freefilename()
#include "../third_party/geometric.h"
#include "../third_party/glwin.h"    // class to generate a window 10 api
#include "../third_party/misc_gl.h"  // connected with previous glwin.h class
#include "../include/extra/gl_gui.h"
#include "../third_party/cnn.h"
#include "../include/extra/tcpsocket.h"

using namespace std;

std::default_random_engine rng;   //This is a random number engine class that generates pseudo-random numbers
static int uniform_rand(int mn, int mx) { return std::uniform_int_distribution<int>(mn, mx)(rng); } //Produces random integer values i, uniformly distributed on the closed interval[mn , mx],

void compress(Frame& frame)   // takes a frame of data and keeps only the relevant 64x64 segmented hand needed for training
{
    if (frame.depth.dim() == int2(64, 64) || frame.depth.dim().x <= 64)
        return; // already compressed to segment
    frame.rgb = Image<byte3>(int2(0, 0));
    frame.fisheye = Image<unsigned char>(int2(0, 0));
    auto segment = HandSegmentVR(frame.depth, 0xF, { 0.1f,0.70f });
    if (frame.ir.raster.size())
    {
        frame.ir = Sample(frame.ir, segment.cam);
        frame.ir.cam.pose = Pose();
    }
    for (auto& p : frame.startpose)
        p = segment.cam.pose.inverse() * p;
    for (auto& p : frame.pose)
        p = segment.cam.pose.inverse() * p;
    segment.cam.pose = Pose();
    frame.mplane = { 0,0,-1,FLT_MAX };
    frame.depth = segment;
}

CNN baby_gestures_cnn()
{
    CNN cnn({});
    //cnn.layers.push_back(new CNN::LFull(120, 128));
    //cnn.layers.push_back(new CNN::LActivation<TanH>(128));

    cnn.layers.push_back(new CNN::LFull(120, 32));
    cnn.layers.push_back(new CNN::LActivation<TanH>(32));
    cnn.layers.push_back(new CNN::LFull(32, 6));
    cnn.layers.push_back(new CNN::LSoftMax(6));
    cnn.Init(); // initializes weights
    return cnn;
}

int main(int argc, char* argv[]) 
{
    CNN cnn2 = baby_gestures_cnn(); 

    cnn2.loadb("../Train-Classifier/HGR_0123(5)cat_2fullylayers.cnnb"); //load CNN that is going to be used at first

    int currentframe = 0;
    int prevframe = -1;

    std::vector<float> errorhistory(128, 1.0f);
    std::vector<float> errorhistory2;

    std::vector<int> categories(6, 0);
    float3 catcolors[] = { {0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1} };

    // Second training trial 
    std::vector<std::vector<float>> samples;
    std::vector<std::vector<float>> labels;
    vector<float> sample_in;

    /************LOADING DATASET ROUTES*****************/

    string fist = "../datasets/_Puño cerrado/hand_data_1";
    string closed_palm = "../datasets/_Palma dedos juntos/hand_data_0";
    string opened_palm = "../datasets/_Palma dedos separados/hand_data_1";
    string wrist_flexion1 = "../datasets/_Flexion muñeca/hand_data_0";
    string wrist_flexion2 = "../datasets/_Flexion muñeca/hand_data_1";
    string wrist_extension = "../datasets/_Extension muñeca/hand_data_0";
    string radial_deviation = "../datasets/_Abduccion muneca (hacia menique)/hand_data_0";
    string ulnar_deviation = "../datasets/_Abduccion muneca (hacia pulgar)/hand_data_0";
    string rock = "../datasets/_Rock/hand_data_0";

    //htk.handmodel.rigidbodies.size() = 17;

    /************SELECTION OF DATASETS WITH POSE INFO FOR TRAINING*****************/
    std::vector<Frame> frames = load_dataset(fist, 17, compress);
    std::vector<Frame> frames2 = load_dataset(closed_palm, 17, compress);
    std::vector<Frame> frames3 = load_dataset(opened_palm, 17, compress);
    //std::vector<Frame> frames3_1 = load_dataset(rock, 17, compress);
    std::vector<Frame> frames4 = load_dataset(wrist_flexion1, 17, compress);
    //std::vector<Frame> frames4_1 = load_dataset(wrist_flexion2, 17);
    //frames4.insert(frames4.end(), std::begin(frames4_1), std::end(frames4_1));
    //std::vector<Frame> frames5 = load_dataset(wrist_extension, 17, compress);
    //std::vector<Frame> frames6 = load_dataset(radial_deviation, 17, compress);
    //std::vector<Frame> frames7 = load_dataset(ulnar_deviation, 17, compress);

    /************DATASETS PROCESSING INTO SIMPLER VECTOR THAT CNN CAN ANALYSE*****************/

    //************CATEGORY 0 - Closed Fist
    for (int i = 0; i < frames.size(); i++)
    {
        auto& current_frame = frames[i];

        for (int i = 0; i < (int)current_frame.pose.size(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                sample_in.push_back(current_frame.pose[i].position[j]);
            }

            for (int j = 0; j < 4; j++)
            {
                sample_in.push_back(current_frame.pose[i].orientation[j]);
            }
        }
        //sample_in.insert(sample_in.end(), begin(completesamples), end(completesamples));
        sample_in.push_back(0); // A 0 needs to be added to the vector to complete the input size of 120
        samples.push_back(sample_in);
        labels.push_back(std::vector<float>(categories.size(), 0.0f));
        labels.back()[0] = 1.0f; // 0 is the number of the category or output node
        sample_in = vector<float>();

    }


    //************CATEGORY 1 - closed palm 
    for (int i = 0; i < frames2.size(); i++)
    {
        auto& current_frame = frames2[i];

        for (int i = 0; i < (int)current_frame.pose.size(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                sample_in.push_back(current_frame.pose[i].position[j]);
            }

            for (int j = 0; j < 4; j++)
            {
                sample_in.push_back(current_frame.pose[i].orientation[j]);
            }
        }
        sample_in.push_back(0);

        samples.push_back(sample_in);
        labels.push_back(std::vector<float>(categories.size(), 0.0f));
        labels.back()[1] = 1.0f; // 1 is the number of the corresponding category or output node
        sample_in = vector<float>();

    }


    //************CATEGORY 2 -  openned palm
    for (int i = 0; i < frames3.size(); i++)
    {
        auto& current_frame = frames3[i];

        for (int i = 0; i < (int)current_frame.pose.size(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                sample_in.push_back(current_frame.pose[i].position[j]);
            }

            for (int j = 0; j < 4; j++)
            {
                sample_in.push_back(current_frame.pose[i].orientation[j]);
            }
        }

        sample_in.push_back(0);
        samples.push_back(sample_in);
        labels.push_back(std::vector<float>(categories.size(), 0.0f));
        labels.back()[2] = 1.0f;
        sample_in = vector<float>();

    }

    //************CATEGORY 3 - wrist flexion
    for (int i = 0; i < frames4.size(); i++)
    {
        auto& current_frame = frames4[i];

        for (int i = 0; i < (int)current_frame.pose.size(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                sample_in.push_back(current_frame.pose[i].position[j]);
            }

            for (int j = 0; j < 4; j++)
            {
                sample_in.push_back(current_frame.pose[i].orientation[j]);
            }
        }

        sample_in.push_back(0);
        samples.push_back(sample_in);
        labels.push_back(std::vector<float>(categories.size(), 0.0f));
        labels.back()[3] = 1.0f;
        sample_in = vector<float>();
    }
    /*
    //************CATEGORY 6 -  rock
    for (int i = 0; i < frames3_1.size(); i++)
    {
        auto& current_frame = frames3_1[i];

        for (int i = 0; i < (int)current_frame.pose.size(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                sample_in.push_back(current_frame.pose[i].position[j]);
            }

            for (int j = 0; j < 4; j++)
            {
                sample_in.push_back(current_frame.pose[i].orientation[j]);
            }
        }

        sample_in.push_back(0);
        samples.push_back(sample_in);
        labels.push_back(std::vector<float>(categories.size(), 0.0f));
        labels.back()[5] = 1.0f;
        sample_in = vector<float>();

    }
    
    //************CATEGORY 4 - wrist extension
    for (int i = 0; i < frames5.size(); i++)
    {
        auto& current_frame = frames5[i];

        for (int i = 0; i < (int)current_frame.pose.size(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                sample_in.push_back(current_frame.pose[i].position[j]);
            }

            for (int j = 0; j < 4; j++)
            {
                sample_in.push_back(current_frame.pose[i].orientation[j]);
            }
        }

        sample_in.push_back(0);
        samples.push_back(sample_in);
        labels.push_back(std::vector<float>(categories.size(), 0.0f));
        labels.back()[4] = 1.0f;
        sample_in = vector<float>();
    }


    //************CATEGORY 5 - radial deviation
    for (int i = 0; i < frames6.size(); i++)
    {
        auto& current_frame = frames6[i];

        for (int i = 0; i < (int)current_frame.pose.size(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                sample_in.push_back(current_frame.pose[i].position[j]);
            }

            for (int j = 0; j < 4; j++)
            {
                sample_in.push_back(current_frame.pose[i].orientation[j]);
            }
        }

        sample_in.push_back(0);
        samples.push_back(sample_in);
        labels.push_back(std::vector<float>(categories.size(), 0.0f));
        labels.back()[5] = 1.0f;
        sample_in = vector<float>();
    }
    */

   // while (glwin.WindowUp())
    {

        // CNN Result:     

        int sample_frame = 1;

        while (sample_frame < samples.size())
        {         
            auto cnn_out = cnn2.Eval(samples[sample_frame]);
            // __int64 cycles_fprop = __rdtsc() - timestart;

            int best = std::max_element(cnn_out.begin(), cnn_out.end()) - cnn_out.begin();
            int category = std::max_element(labels[sample_frame].begin(), labels[sample_frame].end()) - labels[sample_frame].begin();

            for (int i = 0; i < 6; i++)
            {
                cout << cnn_out[i] << "\t";
            }
            cout << endl << "Frame number:\t" << sample_frame << "  //  Predicted category:\t" << best << " // Corresponding category:\t" << category << endl << endl;
            sample_frame += 50;
        }

    
        //glwin.PrintString({ 1,8 }, (category == best) ? "Success, CNN output matches this category" : "Fail, sample miscategorized");

        
        /*
        if (icons[c].raster.size() <= 1)icons[c] = sample_cl;
        snapshots.push_back(sample_cl);
        categories[c]++;
        samples.push_back(sample_in);
        labels.push_back(std::vector<float>(categories.size(), 0.0f));
        labels.back()[c] = 1.0f;
        */
      
    }

    return 0;
}
