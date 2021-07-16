//
//  minimal sample implementation of hand tracking from depth camera.
//
//  The implementation here just uses simple win32 setup and gl wrapper with some basic convenience functions.
//  In particular, just uses opengl 1.1 without any toolkits or dependencies.
//  one ~100 line cpp only,  various support functionality included inline in header files.
//
//  As described in the readme file, there is no sophistication in the segmentation so use this program  
//  with just right hand in the depth camera view volume.
//  Note that results will vary from user to user based on how well it matches the hand model being used and 
//  the hands used in the datasets used to train the CNN.   
//
//  There are some runtime settings that will affect the tracking results, some are exposed as gui widgets.
//  For slower motion when the fingers are extended, its often best to only use the cnn results when 
//  there is a measurable better fit of the geometry to the point cloud.
//  for faster motions, or when there is less geometric features to track (such as a clenched rolling fist), it is usually better
//  to trust the cnn output. 
//

#include <cctype>         // std::tolower
#include <iostream>
#include <iomanip>

#pragma comment (lib, "realsense2.lib")

#include "../third_party/geometric.h"    // also includes linalg.h and uses hlsl aliases
#include "../third_party/mesh.h"         // a simple mesh class with vertex format structure (not tied to dx or opengl)
#include "../third_party/glwin.h"        // does the win32 and opengl setup,  similar to glut or glfw,  header file only implementation
#include "../third_party/misc_gl.h"      // simple mesh drawing, and to draw a scene (camera pose and array of meshes)
#include "../third_party/json.h"    
#include "../include/handtrack.h"        // HandTracker - the system for tracking the hand including eval of cnn in separate thread, physics update, model loading
#include "../include/dcam.h"             // wraps librealsense

#ifdef _DEBUG
#pragma message ("ATTENTION:  Switch To RELEASE Mode!!   Debug is really slow!!")
#endif

using namespace std;

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

int main(int argc, char *argv[]) try
{
	GLWin glwin("htk - testing hand tracking system  using realsense depth camera input",640,360); // Set to a smaller window size 1280 720
	RSCam dcam; // declaration of the realsense camera 
	bool useLeft = false; // Added condition for left hand

	//CLASSIFIER:
	CNN cnn2 = baby_gestures_cnn();
	cnn2.loadb("../Classifier/HandGestureRecognition0.cnnb"); //load CNN that is going to be used at first
	vector<float> cnn_input;

	if (argc == 3)
	{
		dcam.Init(argc==2?argv[1]:""); // Second param is the location of the file
		useLeft = true; // Third param for indicating using left hand
	}
	else if (argc == 2)
	{
		if (*argv[1] == 'l') // User enter key 'l' (letter L) to indicate using left hand 
		{
			dcam.Init();
			useLeft = true;
		}
		else 
		{
			dcam.Init(argc==2?argv[1]:""); // User only enter location of file
		}
	}
	else
	{
		dcam.Init();
	}
	HandTracker htk(useLeft);
	htk.always_take_cnn = false;  // when false the system will just use frame-to-frame when cnn result is not more accurate
	
	glwin.keyboardfunc = [&](int key, int, int)
	{
		switch (std::tolower(key))
		{
		case 'q': case 27:  exit(0); break;  // ESC
		case '+': case '=': htk.scale(1.02f       );  std::cout << "segment_scale " << htk.segment_scale << std::endl; break;  // make model hand larger
		case '-': case '_': htk.scale(1.0f / 1.02f);  std::cout << "segment_scale " << htk.segment_scale << std::endl; break;  // make model hand smaller
		case 'c': htk.always_take_cnn = !htk.always_take_cnn; break;
		case 'a': htk.angles_only     = !htk.angles_only;     break;
		default: std::cerr << "unused key " << (char)key << std::endl; break;
		}
	};
	htk.load_config( "../config.json");

	std::cout << htk.handmodel.rigidbodies.size() << std::endl;

	std::cout << "Valores angulo de cada dedo en salida CNN  0 is open, 3.14 (180 degrees) is clenched." << std::endl;

	std::cout << "Valores angulos de euler de la muñeca respecto al ordenador: Wristroll, pitch , tilt (yaw) y el angulo del primer dedo con respecto a la palma" << std::endl;

	while (glwin.WindowUp())
	{
		auto dimage     = dcam.GetDepth(); //Capturamos la imagen usando la camara inicializada
		auto dimage_rgb = Transform(dimage, [&](unsigned short d) {return byte3((unsigned char)clamp((int)(256 * (1.0f - (d*dimage.cam.depth_scale ) / htk.drangey)), 0, 255)); });
		auto ccolor = Image<byte3>({ dcam.dim() });
		if(dcam.dev) ccolor = Image<byte3>({ dcam.dim() }, (const byte3*)dcam.GetColorFrame().get_data());  // color data from camera

		// Possible to increase the max range from 0.7m to 1.5m for D400
		auto dmesh  = DepthMesh(dimage, { 0.1f,0.7f }, 0.015f, 2);  // Segmentation create points and triangle list from the depth data
		auto dxmesh = MeshSmoothish(dmesh.first, dmesh.second);    // from 3d points to vertices will all the usual attributes
		for (auto &v : dxmesh.verts)
			v.texcoord = dimage.cam.projectz(v.position) / float2(dimage.cam.dim());
		//dxmesh.pose.position.x = 0.15f; // offset 15cm to the side   note this may reveal make the depth mesh look a bit more jagged due to z noise

		htk.update(std::move(dimage));   // update the hand tracking with the current depth camera input
										 // Constructores "move" son usados en lugar de copy constructors en c++

		//CLASSIFIER:

		for (int i = 0; i < htk.handmodel.GetPose().size(); i++)
		{
			for (int j = 0; j < 3; j++)
			{
				cnn_input.push_back(htk.handmodel.GetPose()[i].position[j]);
			}

			for (int j = 0; j < 4; j++)
			{
				cnn_input.push_back(htk.handmodel.GetPose()[i].orientation[j]);
			}
			cnn_input.push_back(0);
		}

		auto cnn_out = cnn2.Eval(cnn_input);
		// __int64 cycles_fprop = __rdtsc() - timestart;

		cnn_input = vector<float>();

		int best = std::max_element(cnn_out.begin(), cnn_out.end()) - cnn_out.begin();
		//int category = std::max_element(labels2[sample_frame].begin(), labels2[sample_frame].end()) - labels2[sample_frame].begin();


		//std::cout << htk.handmodel.GetPose()[0].position << htk.handmodel.GetPose()[0].orientation;

		for (int i = 0; i < 6; i++)
		{
			cout << setprecision(3) << cnn_out[i] << "\t";
		}
		cout << " //  Predicted category: " << best << "\r";

		// OPEN GL :

		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glViewport(0, 0, glwin.res.x, glwin.res.y);
		glClearColor(0.1f+0.2f*(htk.initializing==50) , 0.1f+0.1f*(htk.initializing>0), 0.15f, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		{
			auto allmeshes = Addresses(htk.handmodel.sdmeshes);
			allmeshes.push_back(&dxmesh);
			// render scene with camera near origin but rolled 180 on x to align CV convention with GL convention
			render_scene({ { 0, 0, -0.05f }, normalize(float4(1, 0, 0, 0)) }, allmeshes);  
			drawimage(dimage_rgb              , { 0.01f,0.21f }, { 0.2f ,-0.2f });  // show depth feed from camera
			drawimage(ccolor                  , { 0.01f,0.42f }, { 0.2f ,-0.2f });  // show color feed from camera
			drawimage(htk.get_cnn_difference(), { 0.01f,0.63f }, { 0.15f,-0.2f });  // show segment sent to cnn 
		}
		float segment_scale = htk.segment_scale;
		//Dibujamos los controles habilitados
		WidgetButton(int2{ 3, glwin.res.y - 24 }, int2{ 120,23 }, glwin, "[ESC] quit", []() {exit(0);}).Draw();
		WidgetSwitch(int2{ 3, glwin.res.y - 48 }, int2{ 220,23 }, glwin, htk.always_take_cnn, "[c] cnn priority").Draw();
		WidgetSwitch(int2{ 3, glwin.res.y - 74 }, int2{ 220,23 }, glwin, htk.angles_only    , "[a] cnn angles  ").Draw();
		WidgetSlider(int2{ 3, glwin.res.y - 102}, int2{ 220,23 }, glwin, segment_scale, float2{0.12f ,0.20f }, "[+/-] handsize ").Draw();
		if (segment_scale != htk.segment_scale)
			htk.scale(segment_scale / htk.segment_scale);  // only call the scale routine if its been changed
		glwin.PrintString({2+ 120 / glwin.font_char_dims.x,0 }, "press ESC to quit,  place right hand only in depth camera view, cnn trained for egocentric ");
		glwin.PrintString({2+ 220 / glwin.font_char_dims.x,2 }, "apply cnn %s   ('c' to toggle)", htk.always_take_cnn ? "every time" : "only when ensures closer fit");
		glwin.PrintString({2+ 220 / glwin.font_char_dims.x,4 }, "%s   ('a' to toggle)", htk.angles_only ? "not using depth, just cnn angles" : "using depth for final fit");
		glwin.PrintString({2+ 220 / glwin.font_char_dims.x,6 }, "hand size  %f cm   (use +/- to scale)", segment_scale);

		/*
		//if ( htk.cnn_output_analysis.wristroll > 5.00 || htk.cnn_output_analysis.finger_clenched[0] < 2.00 )  wristroll falla mucho
		if ( htk.cnn_output_analysis.finger_clenched[0] < 2.00)
		{
			glwin.PrintString({ 2 + 120 / glwin.font_char_dims.x,10 }, "POSE: HAND OPENNED ");
		}
		else {
			glwin.PrintString({ 2 + 120 / glwin.font_char_dims.x,10 }, "POSE: FIST CLOSED ");
		}
		*/

		glPopAttrib();
		glwin.SwapBuffers();
		
		//std::cout << "Dimensiones entrada" << htk.cnn_input.dim() << std::endl; //son 64*64

		//std::cout << "|| Fingers:\t";

		/*
		for (int i = 0; i < htk.cnn_output_analysis.finger_clenched.size(); i++)
		{
			std::cout << std::setprecision(3) <<  htk.cnn_output_analysis.finger_clenched[i] << "\t";
		}
		
		//std::cout << "|| heatmap peaks \t";
		
		for (int i = 0; i < htk.cnn_output_analysis.crays.size(); i++)
		{
			std::cout << std::setprecision(3) << htk.cnn_output_analysis.crays[i] << "\t";
		}	
		
		//std::cout << "|| Palm orientation\t";

		//std::cout << std::setprecision(3) << htk.cnn_output_analysis.palmq;
		*/

		//std::cout << std::setprecision(3) << htk.cnn_output_analysis.wristroll << "\t" << htk.cnn_output_analysis.pitch << "\t" << htk.cnn_output_analysis.tilt << "\t dedo 1:\t" << htk.cnn_output_analysis.finger_clenched[0]; //<< "\t" << htk.cnn_output_analysis.pitch << "\t" << htk.cnn_output_analysis.tilt

		//std::cout << htk.handmodel.GetPose()[0].position << htk.handmodel.GetPose()[0].orientation;


		
		//std::cout << htk.cnn_output.size(); son 2000 y pico valores

		//std::cout << std::setprecision(3) << "\t" << htk.pose_estimator.get().pose[1].position << "\t" << htk.pose_estimator.get().pose[0].position;
		
		{
			//std::cout << std::setprecision(2) << "\t" << htk.handmodel.GetPose()[0].orientation ; //<< "\t" << htk.handmodel.GetPose()[4].position << "\t" << htk.handmodel.GetPose()[10].position ;
		}


		//std::cout << "numero de joints: " << htk.handmodel.joints.size() << std::endl << std::endl; //son 16

		/* 
		* for (int i = 0; i < htk.cnn_output.size(); i++)
		{
			std::cout << htk.cnn_output[i] << "\t"; 
		}	
		CNN cnn;
		Image<float>       cnn_input;
		std::vector<float> cnn_output;
		CNNOutputAnalysis  cnn_output_analysis;
		std::future< tracking_with_cnn_results > pose_estimator;
		PhysModel handmodel;
		PhysModel othermodel;
		*/
		//       main(){ ...
//          HandTracker my_hand_tracker;
//          while() // main loop
//             ...
//             my_hand_tracker.update(my_depth_data_from_some_camera);
//             ...
//             my_render_function( htk.handmodel.GetPose(), my_hand_rig); 
//       }
	}
}
catch (const char *c)
{
	MessageBoxA(GetActiveWindow(), "FAIL", c, 0);
}
catch (std::exception e)
{
	MessageBoxA(GetActiveWindow(), "FAIL", e.what(), 0);
}
