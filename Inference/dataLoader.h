#ifndef __UTILS_IMAGE_LOADER_H__
#define __UTILS_IMAGE_LOADER_H__

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"

#include "utils/Utils.h"

#include <cstdlib>
#include <memory>
#include <string>

/** Image loader interface */

namespace arm_compute
{
namespace utils
{	
/** Image feeder interface */
class IImageDataFeeder11
{
public:
    /** Virtual base destructor */
    virtual ~IImageDataFeeder11() = default;
    /** Gets a character from an image feed */
    virtual uint8_t get() = 0;
    /** Feed a whole row to a destination pointer
     *
     * @param[out] dst      Destination pointer
     * @param[in]  row_size Row size in terms of bytes
     */
    virtual void get_row(uint8_t *dst, size_t row_size) = 0;
};

class FileImageFeeder11 : public IImageDataFeeder11
{
public:
    /** Default constructor
     *
     * @param[in] fs Image file stream
     */
    FileImageFeeder11(std::ifstream &fs)
        : _fs(fs)
    {
    }
    // Inherited overridden methods
    uint8_t get() override
    {
        return _fs.get();
    }
    void get_row(uint8_t *dst, size_t row_size) override
    {
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        _fs.read(reinterpret_cast<std::fstream::char_type *>(dst), row_size);
    }

private:
    std::ifstream &_fs;
};
/** Memory Image feeder concrete implementation */
class MemoryImageFeeder11 : public IImageDataFeeder11
{
public:
    /** Default constructor
     *
     * @param[in] data Pointer to data
     */
    MemoryImageFeeder11(const uint8_t *data)
        : _data(data)
    {
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MemoryImageFeeder11(const MemoryImageFeeder11 &) = delete;
    /** Default move constructor */
    MemoryImageFeeder11(MemoryImageFeeder11 &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MemoryImageFeeder11 &operator=(const MemoryImageFeeder11 &) = delete;
    /** Default move assignment operator */
    MemoryImageFeeder11 &operator=(MemoryImageFeeder11 &&) = default;
    // Inherited overridden methods
    uint8_t get() override
    {
        return *_data++;
    }
    void get_row(uint8_t *dst, size_t row_size) override
    {
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        memcpy(dst, _data, row_size);
        _data += row_size;
    }

private:
    const uint8_t *_data;
};
	
class IImageLoader11
{
public:
    /** Default Constructor */
    IImageLoader11()
        : _feeder(nullptr), _width(0), _height(0)
    {
    }
    /** Virtual base destructor */
    virtual ~IImageLoader11() = default;
    /** Return the width of the currently open image file. */
    unsigned int width() const
    {
        return _width;
    }
    /** Return the height of the currently open image file. */
    unsigned int height() const
    {
        return _height;
    }
    /** Return true if the image file is currently open */
    virtual bool is_open() = 0;
    /** Open an image file and reads its metadata (Width, height)
     *
     * @param[in] filename File to open
     */
    virtual void open(const std::string &filename) = 0;
    /** Closes an image file */
    virtual void close() = 0;
    /** Initialise an image's metadata with the dimensions of the image file currently open
     *
     * @param[out] image  Image to initialise
     * @param[in]  format Format to use for the image (Must be RGB888 or U8)
     */
    template <typename T>
    void init_image(T &image, Format format)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON(format != Format::RGB888 && format != Format::U8 && format != Format::F32);

		//const TensorShape input_shape(100, 100, 3);
		const TensorShape image_shape(_width, _height, 3);
		TensorInfo image_info(image_shape, 1, DataType::F32);
		image.allocator()->init(image_info);
    }
	
    /** Fill an image with the content of the currently open image file.
     *
     * @note If the image is a CLImage, the function maps and unmaps the image
     *
     * @param[in,out] image Image to fill (Must be allocated, and of matching dimensions with the opened image file).
     */
    template <typename T>
    void fill_image(T &image)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON(image.info()->dimension(0) != _width || image.info()->dimension(1) != _height || image.info()->dimension(2) != 3);
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(&image, Format::U8, Format::RGB888, Format::F32);
        ARM_COMPUTE_ERROR_ON(_feeder.get() == nullptr);
		//size_t stride = 0;
		
        try
        {
            // Map buffer if creating a CLTensor/GCTensor
            //map(image, true);
            // Validate feeding data
            validate_info(image.info());
			
			//Window window;
			//window.set(Window::DimX, Window::Dimension(0, _width, 1));
			//window.set(Window::DimY, Window::Dimension(0, _height, 1));
			//window.set(Window::DimZ, Window::Dimension(0, 3, 1));
			//stride = 4;
			
			//std::cout<<image.info()->strides_in_bytes()[0]<<std::endl;
			//std::cout<<image.info()->strides_in_bytes()[1]<<std::endl;
			//std::cout<<image.info()->strides_in_bytes()[2]<<std::endl;
					
            //Iterator out(&image, window);
			
			unsigned char red   = 0;
			unsigned char green = 0;
			unsigned char blue  = 0;
			
			uint8_t * data_ptr = image.allocator()->data();
			
			for(int i=0; i < _width ; i++)
			{
				for(int j=0; j < _height; j++)
				{
						red   = _feeder->get();
						green = _feeder->get();
						blue  = _feeder->get();
						
						*reinterpret_cast<float *>(data_ptr) = 0.003921f * red;
						data_ptr = data_ptr + 4;
						*reinterpret_cast<float *>(data_ptr) = 0.003921f * green;
						data_ptr = data_ptr + 4;
						*reinterpret_cast<float *>(data_ptr) = 0.003921f * blue;
						data_ptr = data_ptr + 4;
				}
			} 			
        }
        catch(const std::ifstream::failure &e)
        {
            ARM_COMPUTE_ERROR("Loading image file: %s", e.what());
        }
    }
    /** Fill a tensor with 3 planes (one for each channel) with the content of the currently open image file.
     *
     * @note If the image is a CLImage, the function maps and unmaps the image
     *
     * @param[in,out] tensor Tensor with 3 planes to fill (Must be allocated, and of matching dimensions with the opened image). Data types supported: U8/F32
     * @param[in]     bgr    (Optional) Fill the first plane with blue channel (default = false)
     */
    template <typename T>
    void fill_planar_tensor(T &tensor, bool bgr = false)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&tensor, 1, DataType::U8, DataType::F32);

        const DataLayout  data_layout  = tensor.info()->data_layout();
        const TensorShape tensor_shape = tensor.info()->tensor_shape();

        ARM_COMPUTE_UNUSED(tensor_shape);
        ARM_COMPUTE_ERROR_ON(tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH)] != _width);
        ARM_COMPUTE_ERROR_ON(tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT)] != _height);
        ARM_COMPUTE_ERROR_ON(tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL)] != 3);

        ARM_COMPUTE_ERROR_ON(_feeder.get() == nullptr);

        try
        {
            // Map buffer if creating a CLTensor
            map(tensor, true);

            // Validate feeding data
            validate_info(tensor.info());

            // Stride across channels
            size_t stride_z = 0;

            // Iterate through every pixel of the image
            Window window;
            if(data_layout == DataLayout::NCHW)
            {
                window.set(Window::DimX, Window::Dimension(0, _width, 1));
                window.set(Window::DimY, Window::Dimension(0, _height, 1));
                window.set(Window::DimZ, Window::Dimension(0, 1, 1));
                stride_z = tensor.info()->strides_in_bytes()[2];
            }
            else
            {
                window.set(Window::DimX, Window::Dimension(0, 1, 1));
                window.set(Window::DimY, Window::Dimension(0, _width, 1));
                window.set(Window::DimZ, Window::Dimension(0, _height, 1));
                stride_z = tensor.info()->strides_in_bytes()[0];
            }

            Iterator out(&tensor, window);

            unsigned char red   = 0;
            unsigned char green = 0;
            unsigned char blue  = 0;

            execute_window_loop(window, [&](const Coordinates & id)
            {
                red   = _feeder->get();
                green = _feeder->get();
                blue  = _feeder->get();

                switch(tensor.info()->data_type())
                {
                    case DataType::U8:
                    {
                        *(out.ptr() + 0 * stride_z) = bgr ? blue : red;
                        *(out.ptr() + 1 * stride_z) = green;
                        *(out.ptr() + 2 * stride_z) = bgr ? red : blue;
                        break;
                    }
                    case DataType::F32:
                    {
                        *reinterpret_cast<float *>(out.ptr() + 0 * stride_z) = static_cast<float>(bgr ? blue : red);
                        *reinterpret_cast<float *>(out.ptr() + 1 * stride_z) = static_cast<float>(green);
                        *reinterpret_cast<float *>(out.ptr() + 2 * stride_z) = static_cast<float>(bgr ? red : blue);
                        break;
                    }
                    default:
                    {
                        ARM_COMPUTE_ERROR("Unsupported data type");
                    }
                }
            },
            out);

            // Unmap buffer if creating a CLTensor
            unmap(tensor);
        }
        catch(const std::ifstream::failure &e)
        {
            ARM_COMPUTE_ERROR("Loading image file: %s", e.what());
        }
    }

protected:
    /** Validate metadata */
    virtual void validate_info(const ITensorInfo *tensor_info)
    {
    }

protected:
    std::unique_ptr<IImageDataFeeder11> _feeder;
    int                      _width;
    int                      _height;
};
	
class pmLoader : public IImageLoader11
{
public:
    /** Default Constructor */
    pmLoader()
        : IImageLoader11(), _fs()
    {
    }

    // Inherited methods overridden:
    bool is_open() override
    {
        return _fs.is_open();
    }
    void open(const std::string &filename) override
    {
        ARM_COMPUTE_ERROR_ON(is_open());
        try
        {
            _fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            _fs.open(filename, std::ios::in | std::ios::binary);

            unsigned int max_val = 0;
            std::tie(_width, _height, max_val) = parse_ppm_header(_fs);

            ARM_COMPUTE_ERROR_ON_MSG(max_val >= 256, "2 bytes per colour channel not supported in file %s",
                                     filename.c_str());

            _feeder = support::cpp14::make_unique<FileImageFeeder11>(_fs);
        }
        catch(std::runtime_error &e)
        {
            ARM_COMPUTE_ERROR("Accessing %s: %s", filename.c_str(), e.what());
        }
    }
    void close() override
    {
        if(is_open())
        {
            _fs.close();
            _feeder = nullptr;
        }
        ARM_COMPUTE_ERROR_ON(is_open());
    }

protected:
    // Inherited methods overridden:
    void validate_info(const ITensorInfo *tensor_info) override
    {
        // Check if the file is large enough to fill the image
        const size_t current_position = _fs.tellg();
        _fs.seekg(0, std::ios_base::end);
        const size_t end_position = _fs.tellg();
        _fs.seekg(current_position, std::ios_base::beg);

        ARM_COMPUTE_ERROR_ON_MSG((end_position - current_position) < tensor_info->tensor_shape().total_size(),
                                 "Not enough data in file");
        ARM_COMPUTE_UNUSED(end_position);
    }

private:
    std::ifstream _fs;
};



/** Numpy data loader */
class NPLoader
{
public:
    /** Default constructor */
    NPLoader()
        : _fs(), _shape(), _fortran_order(false), _typestring(), _file_layout(DataLayout::NCHW)
    {
    }

    /** Open a NPY file and reads its metadata
     *
     * @param[in] npy_filename File to open
     * @param[in] file_layout  (Optional) Layout in which the weights are stored in the file.
     */
    void open(const std::string &npy_filename, DataLayout file_layout = DataLayout::NCHW)
    {
        ARM_COMPUTE_ERROR_ON(is_open());
        try
        {
            _fs.open(npy_filename, std::ios::in | std::ios::binary);
            //ARM_COMPUTE_EXIT_ON_MSG(!_fs.good(), "Failed to load binary data from %s", npy_filename.c_str());
            _fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            _file_layout = file_layout;
            std::tie(_shape, _fortran_order, _typestring) = parse_npy_header(_fs);
			std::reverse(_shape.begin(), _shape.end());
			//std::cout<<_shape[0]<<std::endl;
			//std::cout<<_shape[1]<<std::endl;
			//std::cout<<_shape[2]<<std::endl;
			//std::cout<<_shape[3]<<std::endl;
        }
        catch(const std::ifstream::failure &e)
        {
            ARM_COMPUTE_ERROR("Accessing %s: %s", npy_filename.c_str(), e.what());
        }
    }
    /** Return true if a NPY file is currently open */
    bool is_open()
    {
        return _fs.is_open();
    }

    /** Return true if a NPY file is in fortran order */
    bool is_fortran()
    {
        return _fortran_order;
    }

    /** Initialise the tensor's metadata with the dimensions of the NPY file currently open
     *
     * @param[out] tensor Tensor to initialise
     * @param[in]  dt     Data type to use for the tensor
     */
    template <typename T>
    void init_tensor(T &tensor, arm_compute::DataType dt)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON(dt != arm_compute::DataType::F32);

        // Use the size of the input NPY tensor
        TensorShape shape;
        shape.set_num_dimensions(_shape.size());
        for(size_t i = 0; i < _shape.size(); ++i)
        {
            size_t src = i;
            if(_fortran_order)
            {
                src = _shape.size() - 1 - i;
            }
            shape.set(i, _shape.at(src));
        }

        arm_compute::TensorInfo tensor_info(shape, 1, dt);
        tensor.allocator()->init(tensor_info);
    }

    /** Fill a tensor with the content of the currently open NPY file.
     *
     * @note If the tensor is a CLTensor, the function maps and unmaps the tensor
     *
     * @param[in,out] tensor Tensor to fill (Must be allocated, and of matching dimensions with the opened NPY).
     */
    template <typename T>
    void fill_tensor(T &tensor)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(&tensor, arm_compute::DataType::QASYMM8, arm_compute::DataType::S32, arm_compute::DataType::F32);
        try
        {
            // Map buffer if creating a CLTensor
            map(tensor, true);

            // Check if the file is large enough to fill the tensor
            const size_t current_position = _fs.tellg();
            _fs.seekg(0, std::ios_base::end);
            const size_t end_position = _fs.tellg();
            _fs.seekg(current_position, std::ios_base::beg);

            ARM_COMPUTE_ERROR_ON_MSG((end_position - current_position) < tensor.info()->tensor_shape().total_size() * tensor.info()->element_size(),
                                     "Not enough data in file");
            ARM_COMPUTE_UNUSED(end_position);

            // Check if the typestring matches the given one
            std::string expect_typestr = get_typestring(tensor.info()->data_type());
            ARM_COMPUTE_ERROR_ON_MSG(_typestring != expect_typestr, "Typestrings mismatch");

            bool are_layouts_different = (_file_layout != tensor.info()->data_layout());
            // Correct dimensions (Needs to match TensorShape dimension corrections)
            if(_shape.size() != tensor.info()->tensor_shape().num_dimensions())
            {
                for(int i = static_cast<int>(_shape.size()) - 1; i > 0; --i)
                {
                    if(_shape[i] == 1)
                    {
                        _shape.pop_back();
                    }
                    else
                    {
                        break;
                    }
                }
            }

            TensorShape                    permuted_shape = tensor.info()->tensor_shape();
            arm_compute::PermutationVector perm;
            if(are_layouts_different && tensor.info()->tensor_shape().num_dimensions() > 2)
            {
                perm                                    = (tensor.info()->data_layout() == arm_compute::DataLayout::NHWC) ? arm_compute::PermutationVector(2U, 0U, 1U) : arm_compute::PermutationVector(1U, 2U, 0U);
                arm_compute::PermutationVector perm_vec = (tensor.info()->data_layout() == arm_compute::DataLayout::NCHW) ? arm_compute::PermutationVector(2U, 0U, 1U) : arm_compute::PermutationVector(1U, 2U, 0U);

                arm_compute::permute(permuted_shape, perm_vec);
            }

            // Validate tensor shape
            ARM_COMPUTE_ERROR_ON_MSG(_shape.size() != tensor.info()->tensor_shape().num_dimensions(), "Tensor ranks mismatch");
            for(size_t i = 0; i < _shape.size(); ++i)
            {
                ARM_COMPUTE_ERROR_ON_MSG(permuted_shape[i] != _shape[i], "Tensor dimensions mismatch");
            }

            switch(tensor.info()->data_type())
            {
                case arm_compute::DataType::QASYMM8:
                case arm_compute::DataType::S32:
                case arm_compute::DataType::F32:
                case arm_compute::DataType::F16:
                {
                    // Read data
                    if(!are_layouts_different && !_fortran_order && tensor.info()->padding().empty())
                    {
                        // If tensor has no padding read directly from stream.
                        _fs.read(reinterpret_cast<char *>(tensor.buffer()), tensor.info()->total_size());
                    }
                    else
                    {
                        // If tensor has padding or is in fortran order accessing tensor elements through execution window.
                        Window             window;
                        const unsigned int num_dims = _shape.size();
                        if(_fortran_order)
                        {
                            for(unsigned int dim = 0; dim < num_dims; dim++)
                            {
                                permuted_shape.set(dim, _shape[num_dims - dim - 1]);
                                perm.set(dim, num_dims - dim - 1);
                            }
                            if(are_layouts_different)
                            {
                                // Permute only if num_dimensions greater than 2
                                if(num_dims > 2)
                                {
                                    if(_file_layout == DataLayout::NHWC) // i.e destination is NCHW --> permute(1,2,0)
                                    {
                                        arm_compute::permute(perm, arm_compute::PermutationVector(1U, 2U, 0U));
                                    }
                                    else
                                    {
                                        arm_compute::permute(perm, arm_compute::PermutationVector(2U, 0U, 1U));
                                    }
                                }
                            }
                        }
                        window.use_tensor_dimensions(permuted_shape);

                        execute_window_loop(window, [&](const Coordinates & id)
                        {
                            Coordinates dst(id);
                            arm_compute::permute(dst, perm);
                            _fs.read(reinterpret_cast<char *>(tensor.ptr_to_element(dst)), tensor.info()->element_size());
                        });
                    }

                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Unsupported data type");
            }

            // Unmap buffer if creating a CLTensor
            unmap(tensor);
        }
        catch(const std::ifstream::failure &e)
        {
            ARM_COMPUTE_ERROR("Loading NPY file: %s", e.what());
        }
    }
	
	void close()
    {
        if(is_open())
        {
            _fs.close();
        }        
    }

private:
    std::ifstream              _fs;
    std::vector<unsigned long> _shape;
    bool                       _fortran_order;
    std::string                _typestring;
    DataLayout                 _file_layout;
};


} // namespace utils
} // namespace arm_compute
#endif
