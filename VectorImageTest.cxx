/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "itkVectorImage.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageRegionSplitterSlowDimension.h"
#include "itkExtractImageFilter.h"
#include "itkRGBPixel.h"
#include "itkRGBAPixel.h"
#include "itkVectorImage.h"
#include "itkVector.h"
#include "itkPoint.h"
#include "itkCovariantVector.h"
#include "itkFixedArray.h"
#include "itkArray.h"
#include "itkVariableLengthVector.h"
#include <fstream>
#include "itkPipeline.h"
#include "itkInputImage.h"
#include "itkOutputImage.h"
#include "itkOutputTextStream.h"
#include "itkSupportInputImageTypes.h"
#include "itkCastImageFilter.h"
#include "itkComposeImageFilter.h"

template <typename TImage>
int Compose(itk::wasm::Pipeline &pipeline, itk::wasm::InputImage<TImage> &movingImage, itk::wasm::InputImage<TImage> &fixedImage)
{
  using ImageType = TImage;

  pipeline.get_option("input-image")->required()->type_name("INPUT_IMAGE");
  pipeline.get_option("fixed-image")->required()->type_name("INPUT_IMAGE");

  using PipelineOutputType = typename itk::VectorImage<typename ImageType::PixelType, ImageType::ImageDimension>;

  using OutputImageType = itk::wasm::OutputImage<PipelineOutputType>;
  OutputImageType outputImage;
  pipeline.add_option("output-image", outputImage, "Output image")->required()->type_name("OUTPUT_IMAGE");

  ITK_WASM_PARSE(pipeline);

  using FilterType = itk::ComposeImageFilter<ImageType>;
  auto filter = FilterType::New();
  filter->SetInput(0, movingImage.Get());
  filter->SetInput(1, fixedImage.Get());
  filter->UpdateOutputInformation();

  filter->Update();

  typename PipelineOutputType::Pointer vectorImage = filter->GetOutput();
  std::cout << "output components count " << vectorImage->GetNumberOfComponentsPerPixel() << std::endl;

  outputImage.Set(vectorImage);

  return EXIT_SUCCESS;
}
template <typename TImage>
class PipelineFunctor
{
public:
  int operator()(itk::wasm::Pipeline &pipeline)
  {
    using ImageType = TImage;

    using InputImageType = itk::wasm::InputImage<ImageType>;
    InputImageType inputImage;
    pipeline.add_option("input-image", inputImage, "Input image");

    InputImageType fixedImage;
    pipeline.add_option("fixed-image", fixedImage, "Fixed image");

    ITK_WASM_PRE_PARSE(pipeline);

    return Compose<ImageType>(pipeline, inputImage, fixedImage);
  }
};

int main(int argc, char *argv[])
{
  itk::wasm::Pipeline pipeline("ResampleLabelImage", "Resample a label image", argc, argv);

  return itk::wasm::SupportInputImageTypes<PipelineFunctor,
                                           uint8_t,
                                           int8_t,
                                           uint16_t,
                                           int16_t,
                                           float,
                                           double>::Dimensions<2U, 3U>("InputImage", pipeline);
}
