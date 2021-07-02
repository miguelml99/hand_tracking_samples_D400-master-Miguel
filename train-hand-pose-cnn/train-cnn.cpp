// 
// Hand pose example CNN backprop training
//
// CNN example uses heatmap feature map location for output
// Depth camera input and 3D articulated model for ground truth pose information
// Pose tweaking facilities make use of fitting a 3d model to point cloud 
// Given model 3D pose any preferred landmark or information label set is trivially derived
//  
// usage (view multiple files and train):
//   cd ../datasets/
//   ../hposefix/hposefix___.exe  *.rs
//
//

#include <exception>  // base class for standard expceptions (  try throw catch )
#include <iostream>
#include <fstream>
#include <cctype>    // std::tolower converts uppercase letter to lowercase
#include <future>   //Header with facilities that allow asynchronous access to values set by specific providers, possibly in a different thread
#include <sstream>  // Stream class to operate on strings. In C++ stream refers to the stream of characters that are transferred between the program thread and i/o.

#include "../third_party/glwin.h"    // class to generate a window 10 api
#include "../third_party/misc_gl.h"  // connected with previous glwin.h class
#include "../include/physmodel.h"    //simulating physically articulated skeletal models and point cloud fitting
#include "../include/handtrack.h"  //articulated hand skeletal model tracking system
#include "../include/dataset.h"     //has the support code for serializing depth/ir data and the ground truth pose information

std::default_random_engine rng;   //This is a random number engine class that generates pseudo-random numbers
static int uniform_rand(int mn, int mx) { return std::uniform_int_distribution<int>(mn, mx)(rng); } //Produces random integer values i, uniformly distributed on the closed interval[mn , mx],

void compress(Frame &frame)   // takes a frame of data and keeps only the relevant 64x64 segmented hand needed for training
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
	for (auto &p : frame.startpose)
		p = segment.cam.pose.inverse() * p;
	for (auto &p : frame.pose)
		p = segment.cam.pose.inverse() * p;
	segment.cam.pose = Pose();
	frame.mplane = { 0,0,-1,FLT_MAX };
	frame.depth = segment;
}



int main(int argc, const char **argv) try
{
	std::cout << "Example CNN training program with heatmap visualizations.\nIt will likely take a very long time (hours) before seeing  results\n";

	bool useLeft = argc >= 3 ? true : false; // Added condition for left hand

	HandTracker htk(useLeft); //definition of the hand, aqui ya seleccionamos el .cnnb para empezar (seleccionado por defecto)
	float2 drange = { 0.20f, htk.drangey }; // depth range set from 0.2 to 0.7 (default) in meters   

	std::future<std::vector<Frame>> thefuture;
	std::vector<std::string> things_yet_to_load;

	htk.cnn.loadb("../train-hand-pose-cnn/handposeddPruebaPalmayPuño3.cnnb"); //carga la CNN con la que vamos a empezar 
	/*
	for (int i = 1; i < argc; i++)
		if (filesuffix(argv[i]) == ".rs" || filesuffix(argv[i]) == ".ir") // si son archivos .rs o .ir, entendemos que aun habra que cargarlos
			things_yet_to_load.push_back(basepathname(argv[i]));
		else if (filesuffix(argv[i]) == ".cnnb") // si lo que le pasamos es una cnn (pretrained cnn) la cargamos directamente
			htk.cnn.loadb(argv[i]);
	*/

	if (!things_yet_to_load.size()) // comprueba si le hemos metido algo antes- Pero si hubiera cnn cargada anteriormente, no podria volverse a entrenar?
	{
		things_yet_to_load.push_back("../datasets/_Palma dedos juntos/hand_data_0"); //Miguel: here we write the route of the data we are using to build the cnn
		things_yet_to_load.push_back("../datasets/_Palma dedos juntos/hand_data_1"); //Miguel: here we write the route of the data we are using to build the cnn
		things_yet_to_load.push_back("../datasets/_Palma dedos juntos/hand_data_2"); //Miguel: here we write the route of the data we are using to build the cnn
		things_yet_to_load.push_back("../datasets/_Puño cerrado/hand_data_0"); //podemos concatenar datasets para el entrenamiento
		things_yet_to_load.push_back("../datasets/_Puño cerrado/hand_data_1"); //podemos concatenar datasets para el entrenamiento
		things_yet_to_load.push_back("../datasets/_Puño cerrado/hand_data_2"); //podemos concatenar datasets para el entrenamiento
		things_yet_to_load.push_back("../datasets/_Puño cerrado/hand_data_3"); //podemos concatenar datasets para el entrenamiento
	}

	std::string firstname = things_yet_to_load.back();
	things_yet_to_load.pop_back();
	//std::vector<Frame> frames = load_dataset(firstname, (int)htk.handmodel.rigidbodies.size(), compress);  //Seleccionas el dataset que quieres cargar para el entrenamiento
	
	// Si queremos meter varios datasets al mismo tiempo:
	std::vector<Frame> frames1 = load_dataset(firstname, (int)htk.handmodel.rigidbodies.size(),  compress );  //Seleccionas el dataset que quieres cargar para el entrenamiento

	std::string secondname = things_yet_to_load.back();
	things_yet_to_load.pop_back();
	std::vector<Frame> frames2 = load_dataset(secondname, (int)htk.handmodel.rigidbodies.size(), compress);  //Seleccionas el dataset que quieres cargar para el entrenamiento

	std::string thirdname = things_yet_to_load.back();
	things_yet_to_load.pop_back();
	std::vector<Frame> frames3 = load_dataset(thirdname, (int)htk.handmodel.rigidbodies.size(), compress);  //Seleccionas el dataset que quieres cargar para el entrenamiento

	std::string fourthname = things_yet_to_load.back();
	things_yet_to_load.pop_back();
	std::vector<Frame> frames4 = load_dataset(fourthname, (int)htk.handmodel.rigidbodies.size(), compress);

	std::string fifthname = things_yet_to_load.back();
	things_yet_to_load.pop_back();
	std::vector<Frame> frames5 = load_dataset(fifthname, (int)htk.handmodel.rigidbodies.size(), compress);

	std::string sixthname = things_yet_to_load.back();
	things_yet_to_load.pop_back();
	std::vector<Frame> frames6 = load_dataset(sixthname, (int)htk.handmodel.rigidbodies.size(), compress);

	std::string seventhname = things_yet_to_load.back();
	things_yet_to_load.pop_back();
	std::vector<Frame> frames7 = load_dataset(seventhname, (int)htk.handmodel.rigidbodies.size(), compress);

	std::vector<Frame> frames(std::begin(frames1), std::end(frames1));
	frames.insert(frames.end(), std::begin(frames2), std::end(frames2));
	frames.insert(frames.end(), std::begin(frames3), std::end(frames3));
	frames.insert(frames.end(), std::begin(frames4), std::end(frames4));
	frames.insert(frames.end(), std::begin(frames5), std::end(frames5));
	frames.insert(frames.end(), std::begin(frames6), std::end(frames6));
	frames.insert(frames.end(), std::begin(frames7), std::end(frames7));

	
	// have to have at least some data loaded before gui starts

	// GUI
	int currentframe = 0;
	int prevframe = -1;
	GLWin glwin("train-cnn - Example backprop training.", 250+200, 800);

	Pose camera({ 0, 0, -0.10f }, { 1, 0, 0, 0 });  // the viewing opengl camera for rendering the 3D scene consisting of point cloud and posed hand model
	float focuszoffset = 0.5f;  // we pivot the viewing camera about a point half a meter in front of the origin (depth camera location)

	bool   trainmode = false;
	int    train_count = 0;      // how many bprop iterations since last save
	
	auto drawimagevp = [&](const Image<byte3> &image, int2 c, int2 d, std::string caption = "")
	{
		glViewport(c.x, c.y, d.x, d.y);
		drawimage(image, float2(0, 1.0f), float2(1.0f, -1.0f));   // note the flip on vertical 
		int k = 0;
		if (caption.length()) for (auto s : split(caption, "\n")) glwin.PrintString({ 0,k++ }, s.c_str());
	};

	glwin.keydownfunc = [&](int vkey)  // for keys that aren't WM_CHAR events
	{
		// std::cout << "key "<<  vkey << std::endl;
		currentframe = clamp(currentframe + (vkey == VK_RIGHT) - (vkey == VK_LEFT) + (vkey == VK_UP) * 10 - (vkey == VK_DOWN) * 10, 0, (int)frames.size() - 1);
		if (vkey == VK_HOME) currentframe = 0;
		if (vkey == VK_END) currentframe = (int)frames.size() - 1;
	};
	glwin.keyboardfunc = [&](unsigned char key, int x, int y)->void
	{
		switch (std::tolower(key))
		{
		case 'q': case 27: exit(0); break;  // ESC is already handled in mswin.h MsgProc
		case 'c': camera = { { 0, 0, 0 },{ 1, 0, 0, 0 } }; break;
		case '\1': currentframe = 0; break; // ctrl-A ??
		case '\5': currentframe = (int)frames.size() - 1; break; // ctrl-e ??
		case ']': currentframe++; break;
		case '[': currentframe--; break;
		case '}': currentframe += 10; break;
		case '{': currentframe -= 10; break;
		case '\x14': std::cout << "saving cnn..."; htk.cnn.saveb("handposeddPruebaPalmayPuñoTodosDatasets.cnnb"); std::cout << " ...done.  file: handposedd.cnnb\n";train_count = 0; break;  // ctrl-t 
		case '\x18': std::cout << "reset cnn\n";  htk.cnn = PoseInitializerCNN(""); prevframe=-1; break;  // ctrl-x 
		case 't': trainmode = !trainmode; break;
		default:
			std::cout << "unassigned key (" << (int)key << "): '" << key << "'\n";
			break;
		}
		currentframe = std::min(std::max(currentframe, 0), (int)frames.size() - 1);
	};
	while (glwin.WindowUp())
	{
		if (things_yet_to_load.size() && !thefuture.valid())
		{
			std::string fname = things_yet_to_load.back();
			things_yet_to_load.pop_back();
			thefuture = std::async([fname, &htk]() {return load_dataset(fname, (int)htk.handmodel.rigidbodies.size(), compress); });
		}
		if (thefuture.valid() && thefuture.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready)
		{
			auto moreframes = thefuture.get();
			Append(frames, moreframes);
			thefuture = std::future<std::vector<Frame>>();
			std::cout << things_yet_to_load.size() << " things left to load" << std::endl;
		}

		auto currentframe_slider = WidgetSlider(int2{ 1,1 }, int2{ glwin.res.x - 200 - 2,20 - 2 }, glwin, currentframe, int2{ 0,(int)frames.size() - 1 }, "Frame ");

		if (trainmode && !currentframe_slider.focus())  // jump to a random frame for training
			currentframe = uniform_rand(0, (int)frames.size() - 1) / 2 * 2; // keep it even so that odd ones can be 'testing set'

		auto &frame = frames[currentframe];
		if (prevframe != currentframe)
		{
			htk.handmodel.SetPose(frame.pose);
			htk.update_cnn_model(frame.depth);
			prevframe = currentframe;
		}

		auto &cnn_input = htk.cnn_input;
		auto &cnn_output = htk.cnn_output;
		auto &cnn_output_analysis = htk.cnn_output_analysis;
		auto cnn_labels = GatherHandExpectedCNN(frame.pose, camsub(cnn_input.cam, 4)); // used for making 16x16 heatmaps 

		if (trainmode)
		{
			htk.cnn.Train(cnn_input.raster, cnn_labels.cnn_expected, 0.001f);  // used 0.001 when training from randomly initialized weights to avoid exploding gradient problem, 
			train_count++;
		}

		Widget mainview(int2{ 1 ,20 + 1 }, int2{ glwin.res.x - 200 - 2,250 - 20 - 2 }, glwin);  // subregion of window for 3d viewing 

		if (mainview.focus())   // handle any mouse input within the 3D view   
		{
			auto trackball = qconj(camera.orientation);
			trackball = normalize(qmul(mainview.trackball(), trackball));
			camera.orientation = qconj(trackball);
			camera.position = float3(0, 0, focuszoffset) + qzdir(camera.orientation)*length(camera.position - float3(0, 0, focuszoffset));
			camera.position.z -= focuszoffset;
			camera.position *= powf(1.1f, (float)glwin.mousewheel);
			glwin.mousewheel = 0;
			camera.position.z += focuszoffset;
		}

		auto clearcolor = float3(0.15f, 0.15f + 0.2f*trainmode, 0.15f );
		// Drawing Section 
		glClearColor(clearcolor.x, clearcolor.y, clearcolor.z, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity();
		glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();

		currentframe_slider.Draw();

		mainview.DrawBegin();   // the 3D viewport 
		clearcolor *= 0.75f;
		glClearColor(clearcolor.x, clearcolor.y, clearcolor.z, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		mainview.perspective();
		glMultMatrixf(camera.inverse().matrix());

		{
			glLineWidth(1.0f);
			glPushAttrib(GL_ALL_ATTRIB_BITS);
			for (auto &v : cnn_output_analysis.crays)
			{
				glLineWidth((v.w > 1.0f) ? 3.0f : 1.0f);
				glBegin(GL_LINES);
				glColor3fv(rainbow_colors[&v - &cnn_output_analysis.crays[0]]);
				for (int i : {0, 1})
					glVertex3fv(v.xyz() / dot(v.xyz(), qzdir(cnn_input.cam.pose.orientation))*drange[i]);
				glEnd();
			}
			glPopAttrib();
		}
		drawpoints(PointCloud(frame.depth, { drange.x*0.5f,drange.y*1.25f }),{0,1,0});

		std::vector<float3> kmps;
		for (auto k : handmodelfeaturepoints)
			kmps.push_back(frame.pose[k.bone] * k.offset);
		drawpoints(kmps, { 1, 1, 0 }, 5.0f);

		{
			glPushAttrib(GL_ALL_ATTRIB_BITS);
			glLineWidth(1.0f + 3.0f*mainview.focus());
			glColor3fv({ 0.50f, 0.50f, 0.60f });
			glwirefrustumz(frame.depth.cam.deprojectextents(), { 0.15f,std::max(drange.y,1.0f) });
			glPopAttrib();
		}

		// draw the hand model
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glColor3f(1, 1, 1);
		for (auto &m : htk.handmodel.GetMeshes())   // GetMeshes() member syncs to rigidbody poses
			MeshDraw(m);
		glPopAttrib();

		mainview.ortho();
		glwin.PrintString({ 0, -1 }, "%s:%d",frame.fname.c_str(), frame.fid);
		mainview.DrawEnd();

		{
			auto imgds = Transform(cnn_input, [](float d)->byte3 { return byte3(ToGrayScale(d)); });//  .raster.data(), cnn_input.dim(), { 0.7f,0.48f }, { 0.2f, -0.2f });
			imgds = UpSample(imgds);
			for (unsigned int i = 0; i < handmodelfeaturepoints.size(); i++)
			{
				int2 p0 = linalg::clamp<int2, int2>(int2(imgds.cam.projectz(imgds.cam.pose.inverse()*Skin(htk.handmodel.GetPose().data(), handmodelfeaturepoints[i]))), { 0,0 }, imgds.dim() - int2{ 1,1 });
				int2 p1 = int2(cnn_output_analysis.image_points[i] * 4.0f*2.0f);   // scale since we upsampled the image
				for (int t = 0; t < 32; t++)
					imgds.pixel(p0 + (p1 - p0)*t / 31) = byte3(rainbow_colors[i] * 255.0f);
			}
			drawimagevp(imgds, {  1, 500+1 }, { 250 - 2,250 - 2 }, "depth segment");
		}

		{
			auto imgir = ToRGB(Sample(frame.ir, cnn_input.cam));
			for (unsigned int i = 0; i < handmodelfeaturepoints.size(); i++)
				imgir.pixel(linalg::clamp<int2, int2>(int2(imgir.cam.projectz(imgir.cam.pose.inverse()*Skin(htk.handmodel.GetPose().data(), handmodelfeaturepoints[i]))), { 0,0 }, imgir.dim() - int2{ 1,1 })) = byte3(rainbow_colors[i] * 255.0f);
			drawimagevp(imgir, { 1,250 + 1 }, { 250 - 2, 250 - 2 }, "ir segment");//  .raster.data(), cnn_input.dim(), { 0.7f,0.48f }, { 0.2f, -0.2f });
		}
		int panelx = 0, panelw = glwin.res.x - 100 * 2;
		WidgetSwitch({ panelx + 2  , glwin.res.y - 60  + 1 }, { panelw - 5 -100 , 25 - 2 }, glwin, trainmode, "[t] backprop train").Draw();
		WidgetButton({ panelx + 10  , glwin.res.y - 90 + 1 }, { panelw/2 - 20   , 25 - 2 }, glwin, "[^t] save CNN " , [&]() { glwin.keyboardfunc('\x14', 0, 0);}, train_count>1000).Draw();
		WidgetButton({ panelx +  panelw - 90+2  , glwin.res.y - 60 + 4 }, { 90-4, 25 - 8 }, glwin, "[^x] reset CNN ", [&]() { glwin.keyboardfunc('\x18', 0, 0);}).Draw();

		auto lbl_heatmaps = VisualizeHMaps(cnn_labels.hmaps, cnn_input);
		auto cnn_heatmaps = VisualizeHMaps(cnn_output_analysis.hmaps, cnn_input);
		auto lbl_angles = ToRGB(UpSample(UpSample(UpSample(cnn_labels.vmap))));
		auto cnn_angles = ToRGB(UpSample(UpSample(UpSample(ToGrayScale(cnn_output_analysis.vmap)))));
		drawimagevp(cnn_angles  , { glwin.res.x - 2 * 100 + 1,  0 + 1 }, { 100 - 2,100 - 2 }, " \n \n \n \n cnn\n angles");
		drawimagevp(cnn_heatmaps, { glwin.res.x - 2 * 100 + 1,100 + 1 }, { 100 - 2,glwin.res.y - 100 - 2 }, " cnn out");
		drawimagevp(lbl_angles  , { glwin.res.x - 1 * 100 + 1,  0 + 1 }, { 100 - 2,100 - 2 }, " \n \n \n \n actual\n angles");
		drawimagevp(lbl_heatmaps, { glwin.res.x - 1 * 100 + 1,100 + 1 }, { 100 - 2,glwin.res.y - 100 - 2 }, " labels ");
		glViewport(0, 0, glwin.res.x, glwin.res.y);

		int line_y = 0;
		glwin.PrintString({ 0, 0}, "Mode: %s",  trainmode ? "Backprop" : "Frame Inspection");

		glMatrixMode(GL_PROJECTION); glPopMatrix();
		glMatrixMode(GL_MODELVIEW); glPopMatrix();  //leave things in modelview mode
		glPopAttrib();
		glwin.SwapBuffers();
	}
	if (thefuture.valid())  // cleanup any background loading thread
		thefuture = {};//._Abandon();
	return 0;
}
catch (const char *c)
{
	std::cerr << c << "\n";
	MessageBoxA(GetActiveWindow(), c, "FAIL", 0); 
}
catch (const std::exception & e)
{
	std::cerr << e.what() << "\n";
	MessageBoxA(GetActiveWindow(), e.what(), "FAIL", 0);
}
