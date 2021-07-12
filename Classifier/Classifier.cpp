
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

CNN gestures_cnn()  // probably too big to learn quickly with small ground truth sample size
{
    CNN cnn({});
    cnn.layers.push_back(new CNN::LConv({ 64, 64, 1 }, { 5, 5, 1, 16 }, { 60, 60, 16 }));
    cnn.layers.push_back(new CNN::LActivation<TanH>(60 * 60 * 16));
    cnn.layers.push_back(new CNN::LMaxPool(int3(60, 60, 16)));
    cnn.layers.push_back(new CNN::LMaxPool(int3(30, 30, 16)));
    cnn.layers.push_back(new CNN::LConv({ 15, 15, 16 }, { 4, 4, 16, 64 }, { 12, 12, 64 }));
    cnn.layers.push_back(new CNN::LActivation<TanH>(12 * 12 * 64));
    cnn.layers.push_back(new CNN::LMaxPool(int3(12, 12, 64)));
    cnn.layers.push_back(new CNN::LFull(6 * 6 * 64, 64));
    cnn.layers.push_back(new CNN::LActivation<TanH>(64));
    cnn.layers.push_back(new CNN::LFull(64, 6));
    cnn.layers.push_back(new CNN::LSoftMax(6));
    cnn.Init();
    return cnn;
}

int main(int argc, char* argv[]) 
{
    CNN cnn = gestures_cnn(); 

    cout << "Creating gesture recognition network:\n";

    cout << "training started:" << endl;



    //LOADING DATASET WITH POSE INFO

    std::vector<std::string> things_yet_to_load;

    if (!things_yet_to_load.size()) // Checks if we have loaded something in it before- But if there was a cnn previously loaded, could it be trained again? 
    {                               
        //things_yet_to_load.push_back("../datasets/_Palma dedos juntos/hand_data_0"); //Miguel: here we write the route of the data we are using to build the cnn
        things_yet_to_load.push_back("../datasets/_Palma dedos juntos/hand_data_0"); //Miguel: here we write the route of the data we are using to build the cnn
    }

    std::string firstname = things_yet_to_load.back();
    things_yet_to_load.pop_back();

    //htk.handmodel.rigidbodies.size() = 17;
    std::vector<Frame> frames1 = load_dataset(firstname, 17, compress);  //Seleccionas el dataset que quieres cargar para el entrenamiento

    std::vector<Frame> frames(std::begin(frames1), std::end(frames1));

    //**********hand tracking system training: *********
    int currentframe = 0;
    int prevframe = -1;

    bool   trainmode = false;
    int    train_count = 0;      // how many bprop iterations since last save

    currentframe = uniform_rand(0, (int)frames.size() - 1) / 2 * 2; // keep it even so that odd ones can be 'testing set'

    auto& frame = frames[currentframe];

    if (currentframe != prevframe)
    {
        prevframe = currentframe;
    }

    cout << "Total number of frames in this dataset " << frames.size() << endl ;

    vector<float> cnn_input;

    for (int i=0; i < (int)frame.pose.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cnn_input.push_back(frame.pose[i].position[j]);
        }

        for (int j = 0; j < 4; j++)
        {
            cnn_input.push_back(frame.pose[i].orientation[j]);
        }
    }
    cout << "Number of values in this frame: " << (int)cnn_input.size() << endl;
    cout << "Raw values of frame " << currentframe << " in the dataset: " << endl;

    for (int i = 0; i < (int)cnn_input.size(); i++)
    { 
        cout << cnn_input[i] << "\t"; // just printing in command window the raw data that will be use to train the network 
    }

    std::vector<int> categories(6, 0);
    std::vector<float> labels;
    labels = std::vector<float>(categories.size(), 0.0f);

    // First category:
    labels[0] = 1.0f;
    
    cnn.Train(cnn_input, labels, 0.001f);  // used 0.001 when training from randomly initialized weights to avoid exploding gradient problem, 
    train_count++;

    //*******Simpler classification from depth samples ********

    std::default_random_engine rng;
    //int    training = false; trainmode already defined 
    int    samplereview = 0;
    int    currentsample = 0;
    bool   trainstarted = false;
    bool   sinusoidal = false;
    float  time = 0.0f;

    std::vector<std::vector<float>> samples;
    //std::vector<std::vector<string>> labels;
    std::vector<Image<byte3>>       snapshots;

    //std::vector<int> categories(6, 0);
    float3 catcolors[] = { {0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1} };

    std::vector<Image<byte3>> icons;
    for (auto c : categories)
    {
        //icons.push_back(sample_cl);  Sustituir por algun dibujo de cada gesto
    }

    // add training sample (.pose) method from folders

    /*
    if (icons[c].raster.size() <= 1)icons[c] = sample_cl;
    snapshots.push_back(sample_cl);
    categories[c]++;
    samples.push_back(sample_in);
    labels.push_back(std::vector<float>(categories.size(), 0.0f));
    labels.back()[c] = 1.0f;
    */ 

    // load all method

    //save all method

    float sint = 0.0f;


    // keyboard declaration

    //__rdtsc(); //timestamp de windows

    return 0;
}
