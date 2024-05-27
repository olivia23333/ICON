#version 330 core

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 FragNormal;
layout (location = 2) out vec4 FragDepth;

in vec3 Color;
in vec3 CamNormal;
in vec3 depth;

uniform vec3 SHCoeffs[9];

void evaluateH(vec3 n, out float H[9])
{
    float c1 = 0.429043, c2 = 0.511664,
        c3 = 0.743125, c4 = 0.886227, c5 = 0.247708;

    H[0] = c4;
    H[1] = 2.0 * c2 * n[1];
    H[2] = 2.0 * c2 * n[2];
    H[3] = 2.0 * c2 * n[0];
    H[4] = 2.0 * c1 * n[0] * n[1];
    H[5] = 2.0 * c1 * n[1] * n[2];
    H[6] = c3 * n[2] * n[2] - c5;
    H[7] = 2.0 * c1 * n[2] * n[0];
    H[8] = c1 * (n[0] * n[0] - n[1] * n[1]);
}

vec3 evaluateLightingModel(vec3 normal)
{
    float H[9];
    evaluateH(normal, H);
    vec3 res = vec3(0.0);
    for (int i = 0; i < 9; i++) {
        res += H[i] * SHCoeffs[i];
    }
    return res;
}

// vec3 gammaCorrection(vec3 vec, float g)
// {
//     return vec3(pow(vec.x, 1.0/g), pow(vec.y, 1.0/g), pow(vec.z, 1.0/g));
// }

vec4 gammaCorrection(vec4 vec, float g)
{
    return vec4(pow(vec.x, 1.0/g), pow(vec.y, 1.0/g), pow(vec.z, 1.0/g), vec.w);
}

void main() 
{
    FragColor = vec4(Color,1.0);

    vec3 cam_norm_normalized = normalize(CamNormal);
    vec4 shading = vec4(evaluateLightingModel(cam_norm_normalized), 1.0f);
    shading = gammaCorrection(shading, 2.2);
    vec3 rgb = (cam_norm_normalized + 1.0) / 2.0;
    // FragColor = clamp(FragColor * shading, 0.0, 1.0);
    FragColor = clamp(FragColor, 0.0, 1.0);
	FragNormal = vec4(rgb, 1.0);
    FragDepth = vec4(depth.xyz, 1.0);
}
