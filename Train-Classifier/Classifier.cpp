#include <exception>  // base class for standard expceptions (  try throw catch )
#include <iostream>
#include <fstream>
#include <iomanip>
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
    cnn.layers.push_back(new CNN::LConv({ 12, 10, 1 }, { 3, 3, 1, 16 }, { 8, 6, 16 }));
    cnn.layers.push_back(new CNN::LActivation<TanH>(8 * 6 * 16));
    cnn.layers.push_back(new CNN::LMaxPool(int3(8, 6, 16)));
    cnn.layers.push_back(new CNN::LFull(4 * 3 * 16, 32));
    cnn.layers.push_back(new CNN::LActivation<TanH>(32));
    cnn.layers.push_back(new CNN::LFull(32, 6));
    cnn.layers.push_back(new CNN::LSoftMax(6));
    cnn.Init(); // initializes weights
    return cnn;
}

int main(int argc, char* argv[]) 
{
    CNN cnn = baby_gestures_cnn(); 

    //ofstream error_history ("Error_history.csv"); // CVS file in which we can load the learning curve of the network 

    //cnn.loadb("../Classifier/HandGestureRecognition.cnnb"); //load CNN that is going to be used at first
    GLWin glwin("Gesture recognition system", 300, 200);

    //std::vector<float> completesamples(25, 0);
    std::vector<int> categories(6, 0);

    int currentframe = 0;
    int prevframe = -1;

    bool   trainmode = false;
    int    train_count = 0;      // how many bprop iterations since last save 

    std::default_random_engine rng;

    // Taining data
    vector<vector<float>> samples;
    vector<vector<float>> labels;
    vector<float> sample_in;

    /************LOADING DATASET ROUTES*****************/

    string fist = "../datasets/_Puño cerrado/hand_data_1";
    string closed_palm = "../datasets/_Palma dedos juntos/hand_data_0";
    string opened_palm = "../datasets/_Palma dedos separados/hand_data_0";
    string wrist_flexion = "../datasets/_Flexion muñeca/hand_data_0";
    string wrist_extension = "../datasets/_Extension muñeca/hand_data_0";
    string radial_deviation = "../datasets/_Abduccion muneca (hacia menique)/hand_data_0";
    string ulnar_deviation = "../datasets/_Abduccion muneca (hacia pulgar)/hand_data_0";
    
    //htk.handmodel.rigidbodies.size() = 17;

    /************SELECTION OF DATASETS WITH POSE INFO FOR TRAINING*****************/
    std::vector<Frame> frames = load_dataset(fist, 17, compress); 
    std::vector<Frame> frames2 = load_dataset(closed_palm, 17, compress);
    std::vector<Frame> frames3 = load_dataset(opened_palm, 17, compress);
    //std::vector<Frame> frames4 = load_dataset(wrist_flexion, 17, compress);
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
    
    /*
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
    

    //************CATEGORY 5 - 
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

    //Keyboard implementation
    glwin.keyboardfunc = [&](unsigned char key, int x, int y)->void
    {
        switch (std::tolower(key))
        {
        case 'q': case 27: exit(0); break;  // ESC is already handled in mswin.h MsgProc
        case 's': std::cout << "saving cnn..."; cnn.saveb("HandGestureRecognition_56(0)cat.cnnb"); std::cout << " ...done.  file: handposedd.cnnb\n"; break;  // ctrl-t 
        //case 't': if (key == 'T') trainmode++; else trainmode = (!trainmode) * 5; break; // t toggles while shift-T will continue to increas amount of backprop iterations per frame
        case 't': trainmode = !trainmode; break;
        default:
            std::cout << "unassigned key (" << (int)key << "): '" << key << "'\n";
            break;
        }
        //  currentframe = std::min(std::max(currentframe, 0), (int)frames.size() - 1);
    };

    cout << "Creating gesture recognition network...\n";

    void* current_selection = NULL;

    while (glwin.WindowUp())
    {
        while (currentframe == prevframe){
            currentframe = uniform_rand(0, (int)samples.size() - 1);        
        }
       
        prevframe = currentframe;
        float mse = 0.0f;
        
        if (trainmode)
        {
            currentframe = uniform_rand(0, (int)samples.size() - 1); // keep it even so that odd ones can be 'testing set'

            mse = cnn.Train(samples[currentframe], labels[currentframe], 0.001f);  // used 0.001 when training from randomly initialized weights to avoid exploding gradient problem, 
            train_count++;
            cout << setprecision(3) << " Mean square error:  " << mse << " \t- square root of mse: " << sqrt(mse) << "\r";

            //error_history << sqrt(mse) << endl; esto solo si queremos guardar el avance 
        }

       // Graphical user Interface and GL window:
        GUI gui(glwin, current_selection);
        static int debugpanelheight = 1;
        gui.splity(debugpanelheight, 3);
        glPushAttrib(GL_ALL_ATTRIB_BITS);

        glwin.PrintString({ 1, 2 }, "mse: %5.2f", mse);
        glwin.PrintString({ 1, 4 }, "error: %5.2f", sqrt(mse));
        glwin.PrintString({ 0, 8 }, "Train count: %5.2f", train_count);

        gui.pop();
        // keyboard declaration
        int panelx = 0, panelw = glwin.res.x - 100 * 2;

        WidgetSwitch({ panelx + 2  , glwin.res.y - 60 + 1 }, { panelw - 5 - 100 , 25 - 2 }, glwin, trainmode, "[t] backprop train").Draw();
        WidgetButton({ panelx + 10  , glwin.res.y - 90 + 1 }, { panelw / 2 - 20   , 25 - 2 }, glwin, "[^t] save CNN ", [&]() { glwin.keyboardfunc('s', 0, 0);}, train_count > 1000).Draw();
        
        gui.ortho();
        glColor3f(1, 0, 0);
        glBegin(GL_QUAD_STRIP);
        glEnd();
        glColor3f(1, 1, 1);
        glPopAttrib();
        glwin.SwapBuffers();     
    }
    
    //error_history.close();
    return 0;
}
