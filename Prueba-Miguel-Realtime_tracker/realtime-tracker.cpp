// Prueba_Miguel_0.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <cctype>         // std::tolower
#include <iostream>
#include <cctype>    // std::tolower converts uppercase letter to lowercase

/*
#include <exception>  // base class for standard expceptions (  try throw catch )
#include <fstream>
#include <future>   //Header with facilities that allow asynchronous access to values set by specific providers, possibly in a different thread
#include <sstream>  // Stream class to operate on strings. In C++ stream refers to the stream of characters that are transferred between the program thread and i/o.
*/

#pragma comment (lib, "realsense2.lib")

#include "../third_party/geometric.h"    // also includes linalg.h and uses hlsl aliases
#include "../third_party/mesh.h"         // a simple mesh class with vertex format structure (not tied to dx or opengl)
#include "../third_party/glwin.h"    // class to generate a window 10 api
#include "../third_party/misc_gl.h"  // connected with previous glwin.h class
#include "../third_party/json.h"    
#include "../include/handtrack.h"        // HandTracker - the system for tracking the hand including eval of cnn in separate thread, physics update, model loading
#include "../include/dcam.h"             // wraps librealsense

#ifdef _DEBUG
#pragma message ("ATTENTION:  Switch To RELEASE Mode!!   Debug is really slow!!")
#endif

int main(int argc, char* argv[])try
{

    GLWin glwin("htk - testing hand tracking system  using realsense depth camera input", 700, 400); // nombre de la ventana gl que creamos junto con su tamaño
    RSCam dcam; // declaration of the realsense camera
    bool useLeft = false; // Added condition for left hand

    dcam.Init();

    HandTracker htk(useLeft);
    htk.always_take_cnn = true;  // when false the system will just use frame-to-frame when cnn result is not more accurate

    glwin.keyboardfunc = [&](int key, int, int)  // Teclas que queramos implementar
    {
        switch (std::tolower(key))
        {
        case 'q': case 27:  exit(0); break;  // ESC
        default: std::cerr << "unused key " << (char)key << std::endl; break;
        }
    };

    htk.load_config("../config.json");

    while (glwin.WindowUp())
    {
        auto dimage = dcam.GetDepth(); //Capturamos la imagen usando la camara inicializada
        
        auto dimage_rgb = Transform(dimage, [&](unsigned short d) {return byte3((unsigned char)clamp((int)(256 * (1.0f - (d * dimage.cam.depth_scale) / htk.drangey)), 0, 255)); });
        auto ccolor = Image<byte3>({ dcam.dim() });
        if (dcam.dev) ccolor = Image<byte3>({ dcam.dim() }, (const byte3*)dcam.GetColorFrame().get_data());  // color data from camera


        auto dmesh = DepthMesh(dimage, { 0.1f,0.7f }, 0.015f, 2);  // create points and triangle list from the depth data
        auto dxmesh = MeshSmoothish(dmesh.first, dmesh.second);    // from 3d points to vertices will all the usual attributes
        /*
        for (auto& v : dxmesh.verts)
            v.texcoord = dimage.cam.projectz(v.position) / float2(dimage.cam.dim());
        */

        htk.update(std::move(dimage));   // update the hand tracking with the current depth camera input
                                         // Constructores "move" son usados en lugar de copy constructors en c++

        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glViewport(0, 0, glwin.res.x, glwin.res.y);
        glClearColor(0.1f + 0.2f * (htk.initializing == 50), 0.1f + 0.1f * (htk.initializing > 0), 0.15f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        {
            auto allmeshes = Addresses(htk.handmodel.sdmeshes);
            allmeshes.push_back(&dxmesh);
            // render scene with camera near origin but rolled 180 on x to align CV convention with GL convention
            render_scene({ { 0, 0, -0.05f }, normalize(float4(1, 0, 0, 0)) }, allmeshes);
            //drawimage(dimage_rgb, { 0.01f,0.21f }, { 0.2f ,-0.2f });  // show depth feed from camera
            //drawimage(ccolor, { 0.01f,0.42f }, { 0.2f ,-0.2f });  // show color feed from camera
            //drawimage(htk.get_cnn_difference(), { 0.01f,0.63f }, { 0.15f,-0.2f });  // show segment sent to cnn 
        }
        float segment_scale = htk.segment_scale;

        glwin.PrintString({ 2 + 120 / glwin.font_char_dims.x,0 }, "press ESC to quit,  place right hand only in depth camera view, cnn trained for egocentric ");


        glPopAttrib();
        glwin.SwapBuffers();
    }
}
catch (const char* c)
{
    MessageBoxA(GetActiveWindow(), "FAIL", c, 0);
}
catch (std::exception e)
{
    MessageBoxA(GetActiveWindow(), "FAIL", e.what(), 0);
}

/*
int a = 2;
std::pair <int, int > fpair(7, 8);
fpair.first;
std::cout << "Hello World!\n";
for (int i = 0; i < 4; i++)
{
    cout << i << endl;
    if (a == i)
    {
        cout << "The variable a and the loop counter are both equal to " << a << endl;
        break;
    }
}

std::cout << "Programm ended " << std::endl;
*/



// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
