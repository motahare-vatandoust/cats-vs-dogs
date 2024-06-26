import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cat vs. Dog',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Cat vs. Dog'),
      debugShowCheckedModeBanner: false,
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  bool _loading = true;
  late File _image;
  List<String> _output = [];
  final picker = ImagePicker();
  late Interpreter _interpreter;
  late List<String> _labels;

  @override
  void initState() {
    super.initState();
    loadModel().then((_) {
      setState(() {});
    });
  }

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('tflite_model.tflite');
      _labels = await rootBundle
          .loadString('labels.txt')
          .then((value) => value.split('\n'));
    } on PlatformException {
      print('Failed to load the model.');
    }
  }

  @override
  void dispose() {
    _interpreter.close();
    super.dispose();
  }

  Future<void> classifyImage(File image) async {
    try {
      // Load image using TensorImage
      TensorImage tensorImage = TensorImage.fromFile(image);

      // Create ImageProcessor for preprocessing
      ImageProcessor imageProcessor = ImageProcessorBuilder()
          .add(ResizeOp(224, 224, ResizeMethod.BILINEAR))
          .add(NormalizeOp(0, 255)) // Normalize between 0 and 1 if required by model
          .build();

      // Process the image
      tensorImage = imageProcessor.process(tensorImage);

      // Allocate output tensor
      var outputTensor = TensorBufferFloat([1, _labels.length]);

      // Run inference
      _interpreter.run(tensorImage.buffer, outputTensor.buffer);

      // Get output results
      var results = outputTensor.getDoubleList();

      // Process results
      setState(() {
        _output = getTopResults(results, _labels);
        _loading = false;
      });
    } catch (e) {
      print('Error during image classification: $e');
    }
  }

  List<String> getTopResults(List<double> output, List<String> labels) {
    var results = output.asMap().entries.map((entry) {
      return MapEntry(entry.key, entry.value);
    }).toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    var topResults = results.sublist(0, 2).map((result) {
      return "${labels[result.key]}: ${result.value.toStringAsFixed(3)}";
    }).toList();

    return topResults;
  }

  Future<void> pickImage() async {
    var image = await picker.pickImage(source: ImageSource.camera);
    if (image == null) return;

    setState(() {
      _image = File(image.path);
      _loading = true;
    });

    classifyImage(_image);
  }

  Future<void> pickGalleryImage() async {
    var image = await picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;

    setState(() {
      _image = File(image.path);
      _loading = true;
    });

    classifyImage(_image);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Container(
        padding: EdgeInsets.symmetric(horizontal: 24),
        child: Center(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Center(
                child: _loading
                    ? Container(
                  width: 280,
                  child: Column(
                    children: [
                      Image.asset('assets/home.png'),
                      SizedBox(height: 50),
                    ],
                  ),
                )
                    : Container(
                  child: Column(
                    children: [
                      Container(
                        height: 250,
                        child: Image.file(_image),
                      ),
                      SizedBox(height: 20),
                      _output.isNotEmpty
                          ? Text(
                        '${_output[0]}',
                        style: TextStyle(
                            color: Colors.red, fontSize: 20),
                      )
                          : Container(),
                    ],
                  ),
                ),
              ),
              Container(
                width: MediaQuery.of(context).size.width,
                child: Column(
                  children: [
                    GestureDetector(
                      onTap: pickImage,
                      child: Container(
                        width: MediaQuery.of(context).size.width - 190,
                        alignment: Alignment.center,
                        padding: EdgeInsets.symmetric(horizontal: 24, vertical: 17),
                        decoration: BoxDecoration(
                            color: Color(0xC9EBECFF),
                            borderRadius: BorderRadius.circular(6)),
                        child: Text("Take a photo"),
                      ),
                    ),
                    SizedBox(height: 10),
                    GestureDetector(
                      onTap: pickGalleryImage,
                      child: Container(
                        width: MediaQuery.of(context).size.width - 190,
                        alignment: Alignment.center,
                        padding: EdgeInsets.symmetric(horizontal: 24, vertical: 17),
                        decoration: BoxDecoration(
                            color: Color(0xC9EBECFF),
                            borderRadius: BorderRadius.circular(6)),
                        child: Text("Camera Roll"),
                      ),
                    ),
                  ],
                ),
              )
            ],
          ),
        ),
      ),
    );
  }
}
