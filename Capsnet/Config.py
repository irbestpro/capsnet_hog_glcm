class Config:
    def __init__(self):
            # CNN (cnn)
            self.cnn_in_channels = 1 # For GrayScale
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 56 * 56

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 56 * 56
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder size Equals To Preprocessed Image Size
            self.input_width = 128
            self.input_height = 128
