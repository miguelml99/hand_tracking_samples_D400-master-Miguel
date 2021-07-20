
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
    cnn.layers.push_back(new CNN::LConv({ 12, 10, 1 }, { 5, 5, 1, 16 }, { 8, 6, 16 }));
    cnn.layers.push_back(new CNN::LActivation<TanH>(8 * 6 * 16));
    cnn.layers.push_back(new CNN::LMaxPool(int3(8, 6, 16)));
    cnn.layers.push_back(new CNN::LFull(4 * 3 * 16, 32));
    cnn.layers.push_back(new CNN::LActivation<TanH>(32));
    cnn.layers.push_back(new CNN::LFull(32, 6));
    cnn.layers.push_back(new CNN::LSoftMax(6));
    cnn.Init(); // initializes weights
    return cnn;
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
    CNN cnn = baby_gestures_cnn(); 

    //ofstream error_history ("Error_history.csv");

    //cnn.loadb("../Classifier/HandGestureRecognition.cnnb"); //load CNN that is going to be used at first

    cout << "Creating gesture recognition network:\n";

    cout << "training started:" << endl;

    //LOADING DATASET WITH POSE INFO

    std::vector<std::string> things_yet_to_load;

    if (!things_yet_to_load.size()) // Checks if we have loaded something in it before- But if there was a cnn previously loaded, could it be trained again? 
    {                               
        things_yet_to_load.push_back("../datasets/_Extension muñeca/hand_data_0");
        things_yet_to_load.push_back("../datasets/_Flexion muñeca/hand_data_0");
        things_yet_to_load.push_back("../datasets/_Palma dedos separados/hand_data_0");
        things_yet_to_load.push_back("../datasets/_Palma dedos juntos/hand_data_0"); //Miguel: here we write the route of the data we are using to build the cnn
        things_yet_to_load.push_back("../datasets/_Puño cerrado/hand_data_1"); //podemos concatenar datasets para el entrenamiento
        
    }

    std::string fist = things_yet_to_load.back();
    things_yet_to_load.pop_back();

    std::string closed_palm = things_yet_to_load.back();
    things_yet_to_load.pop_back();

    std::string opened_palm = things_yet_to_load.back();
    things_yet_to_load.pop_back();

    std::string wrist_flexion = things_yet_to_load.back();
    things_yet_to_load.pop_back();

    std::string wrist_extension = things_yet_to_load.back();
    things_yet_to_load.pop_back();

    //htk.handmodel.rigidbodies.size() = 17;
    //std::vector<Frame> frames = load_dataset(firstname, 17, compress);  //Seleccionas el dataset que quieres cargar para el entrenamiento

    std::vector<Frame> frames = load_dataset(fist, 17, compress);  //Seleccionas el dataset que quieres cargar para el entrenamiento
    std::vector<Frame> frames2 = load_dataset(closed_palm, 17, compress);  //Seleccionas el dataset que quieres cargar para el entrenamiento
    std::vector<Frame> frames3 = load_dataset(opened_palm, 17, compress);
    std::vector<Frame> frames4 = load_dataset(wrist_flexion, 17, compress);
    std::vector<Frame> frames5 = load_dataset(wrist_extension, 17, compress);

    GLWin glwin("Gesture recognition system", 1240, 660);

    //**********hand tracking system training: *********
    std::vector<int> categories(6, 0);

    int currentframe = 0;
    int prevframe = -1;

    bool   trainmode = false;
    int    train_count = 0;      // how many bprop iterations since last save 

    cout << "Total number of frames loaded " << frames.size() + frames2.size() + frames3.size() << endl;

    //*******Simpler classification from depth samples  ********

    int frame2 = 0;
    std::vector<float> errorhistory;

    std::default_random_engine rng;
    //int    training = false;//trainmode already defined 
    int    samplereview = 0;
    int    currentsample2 = 0;
    bool   trainstarted = false;
    bool   sinusoidal = false;
    float  time = 0.0f;
    //std::vector<int> categories(6, 0);
    float3 catcolors[] = { {0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1} };

    // Second training trial 
    std::vector<std::vector<float>> samples;
    std::vector<std::vector<float>> labels;
    vector<float> sample_in;
   
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

        sample_in.push_back(0);
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
    
  // for (int i = 0; i < (int)samples2[361].size(); i++)
    {
      //  cout << samples2[361][i] << "\t"; // just printing in command window the raw data that will be use to train the network 
      
    }

    //Keys implementation
    glwin.keyboardfunc = [&](unsigned char key, int x, int y)->void
    {
        switch (std::tolower(key))
        {
        case 'q': case 27: exit(0); break;  // ESC is already handled in mswin.h MsgProc
        case 's': std::cout << "saving cnn..."; cnn.saveb("HandGestureRecognition_013cat.cnnb"); std::cout << " ...done.  file: handposedd.cnnb\n"; break;  // ctrl-t 
        //case 't': if (key == 'T') trainmode++; else trainmode = (!trainmode) * 5; break; // t toggles while shift-T will continue to increas amount of backprop iterations per frame
        case 't': trainmode = !trainmode; break;
        default:
            std::cout << "unassigned key (" << (int)key << "): '" << key << "'\n";
            break;
        }
        //  currentframe = std::min(std::max(currentframe, 0), (int)frames.size() - 1);
    };

    void* current_selection = NULL;
    auto random = 50; // this is the index of the frame that is going to be sent into the cnn after training 

    while (glwin.WindowUp())
    {

        // *******TRAINING BASED ON: Train Hand tracking samples ********

        while (currentframe == prevframe)
        {
            currentframe = uniform_rand(0, (int)samples.size() - 1);        
        }

       
        prevframe = currentframe;

        //cout << "Number of values in this frame: " << (int)samples2[currentframe].size() << " - Frame sent to the cnn " << currentframe << " - train iterations: " << train_count << " \r";

        //cout << currentframe << "\t";
        //for (int i = 0; i < (int)labels2[currentframe].size(); i++)
        {
            //cout << samples2[currentframe][i] << "\t"; // just printing in command window the raw data that will be use to train the network 
            //cout << labels2[currentframe][i] << "\t";
        }
        //cout << endl << endl;
        float mse = 0.0f;
        
        if (trainmode)
        {
            currentframe = uniform_rand(0, (int)samples.size() - 1); // keep it even so that odd ones can be 'testing set'

            mse = cnn.Train(samples[currentframe], labels[currentframe], 0.001f);  // used 0.001 when training from randomly initialized weights to avoid exploding gradient problem, 
            train_count++;
            cout << setprecision(3) << " Mean square error:  " << mse << " \t- square root of mse: " << sqrt(mse) << "\r";
            
            //errorhistory.push_back(sqrt(mse));
            //error_history << sqrt(mse) << endl; esto solo si queremos guardar el avance 
        }
        
        /*
        //*******TRAINING BASED ON: Simpler classification from depth samples ********

        // TRAINING 2.0
        
        //for (int i = 0; i < train_count && samples2.size() != 0; i++)  // why do we use this ??
        {
            trainstarted = true;
            //auto s = std::uniform_int<int>(0, (int)samples2.size() - 1)(rng); // does this random generator discard odd numbers?
            //uniform int produces integer values according to a uniform discrete distribution
            auto s = uniform_rand(0, (int)frames.size() - 1) / 2 * 2; // keep it even so that odd ones can be 'testing set'
            mse = cnn.Train(samples2[s], labels2[s], 0.001f);  // used 0.001 when training from randomly initialized weights to avoid exploding gradient problem, 
            //mse = train_count ? mse / (float)train_count : 0;
            //cout << "Number of values in this frame/sample: " << samples2[s].size() << " - Frame sent to the cnn " << s << " - train iterations: " << train_count << "  \r";
        }
        cout << " Mean square error:  " << mse << "- square root of mse: "<< sqrt(mse) << "\n";
       
        errorhistory[frame2 % (errorhistory.size() / 2)] = errorhistory[frame2 % (errorhistory.size() / 2) + (errorhistory.size() / 2)] = sqrt(mse);

        train_count++;
        frame2++;

        // CNN Result: 
        
        random += 20;
        if (random > samples2.size()) 
        {
            random = 0;
        }
       // auto cnn_out = trainstarted ? cnn.Eval(samples2[random]) : std::vector<float>(categories.size(), 0.0f);
        //__int64 cycles_fprop = __rdtsc() - timestart;
       // int best = std::max_element(cnn_out.begin(), cnn_out.end()) - cnn_out.begin();

       // int category = std::max_element(labels2[random].begin(), labels2[random].end()) - labels2[random].begin();
       */
        glwin.PrintString({ 1, 2 }, "mse: %5.2f", mse);
        glwin.PrintString({ 1, 4 }, "error: %5.2f", sqrt(mse));
        glwin.PrintString({ 0, 8 }, "Train count: %5.2f", train_count);
        

        //glwin.PrintString({ 1, 6 }, "Frame number analized: %5.2f", random);

        /*
        int j = 0;
        
        for (int i : cnn_out)
        {
            glwin.PrintString({ 2, j }, "%5.2f", cnn_out[i]);
            j++;
        }
        */
        
        //glwin.PrintString({ 1,8 }, (category == best) ? "Success, CNN output matches this category" : "Fail, sample miscategorized");

        
        /*
        if (icons[c].raster.size() <= 1)icons[c] = sample_cl;
        snapshots.push_back(sample_cl);
        categories[c]++;
        samples.push_back(sample_in);
        labels.push_back(std::vector<float>(categories.size(), 0.0f));
        labels.back()[c] = 1.0f;
        */

        // keyboard declaration
        int panelx = 0, panelw = glwin.res.x - 100 * 2;

        WidgetSwitch({ panelx + 2  , glwin.res.y - 60 + 1 }, { panelw - 5 - 100 , 25 - 2 }, glwin, trainmode, "[t] backprop train").Draw();
        WidgetButton({ panelx + 10  , glwin.res.y - 90 + 1 }, { panelw / 2 - 20   , 25 - 2 }, glwin, "[^t] save CNN ", [&]() { glwin.keyboardfunc('s', 0, 0);}, train_count > 1000).Draw();
        
        /* Graphical user Interface
        GUI gui(glwin, current_selection);
        gui.ortho();
        glColor3f(1, 0, 0);
        glBegin(GL_QUAD_STRIP);
        for (unsigned int i = 0; i < errorhistory.size() / 2; i++)
        {
            glVertex2f((float)i / (errorhistory.size() / 2), errorhistory[1 + i + frame2 % (errorhistory.size() / 2)]);
            glVertex2f((float)i / (errorhistory.size() / 2), 0);
        }
        glEnd();
        glColor3f(1, 1, 1);
        */

        glPopAttrib();
        glwin.SwapBuffers();     
    }

   // for (int i = 0; i < train_count; i++)
    {
    //    error_history << errorhistory[i] << endl;
    }
    
    //error_history.close();
    return 0;
}
