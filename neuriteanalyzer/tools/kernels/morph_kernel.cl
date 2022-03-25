
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

#define OP_DILATE 0
#define OP_ERODE 1

__kernel void morphologicalProcessing(__read_only image2d_t input, __write_only image2d_t output, int operation, int kernelX, int kernelY)
{
    
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    float extremePxVal = operation == OP_DILATE ? 0.0f : 1.0f;
    float pxVal;
    int kernelRadX = (kernelX-1)/2;
    int kernelRadY = (kernelY-1)/2;
    
    for(int i = -kernelRadX; i <= kernelRadX; ++i)
    {
    
        for(int j = -kernelRadY; j <= kernelRadY; ++j)
        {
        
            pxVal = read_imagef(input,sampler,(int2)(x+i,y+j)).s0;
            
            if(operation == OP_DILATE)
            {
                if(pxVal == 1)
                {
                    break;
                }
            } else
            {
                if(pxVal == 0)
                {
                    break;
                }
            }
                
        
        }
    
        if(operation == OP_DILATE)
        {
            if(pxVal == 1)
            {
                break;
            }
        } else
        {
            if(pxVal == 0)
            {
                break;
            }
        }
    
    }
    
    write_imagef(output,(int2)(x,y),(float4)(extremePxVal,0.0f,0.0f,0.0f));

}
