import os 

def main():
    for mode in ['shading', 'ball']:
        if mode == 'shading':
            modename = ''
        if mode == 'ball':
            modename = "_ball"
        for version in [1,2]:
            for order in [2,3,4,6,10,20,50,100]:
                in_dir = f'/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate/everett_kitchen2/shading_exr_perspective_v3_order{order}_v{version}{modename}_viz_ldr'
                out_file = f'/ist/ist-share/vision/pakkapon/relight/DiffusionLight-SingleLoRA/output/multi_illumination/least_square/rotate/everett_kitchen2/video/v{version}_{mode}_order{order}.mp4'
                if os.path.exists(out_file):
                    continue
                if not os.path.exists(in_dir):
                    continue
                os.system(f'ffmpeg -r 10 -i {in_dir}/dir_%d_mip2.png -c:v libx264 -crf 12 -pix_fmt yuv420p {out_file}')

if __name__ == "__main__":
    main()

"""
ffmpeg \
-i v1_ball_order2.mp4 -i v1_ball_order3.mp4 -i v1_ball_order4.mp4 -i v1_ball_order6.mp4 -i v1_ball_order10.mp4 -i v1_ball_order20.mp4 -i v1_ball_order50.mp4 -i v1_ball_order100.mp4 \
-i v2_ball_order2.mp4 -i v2_ball_order3.mp4 -i v2_ball_order4.mp4 -i v2_ball_order6.mp4 -i v2_ball_order10.mp4 -i v2_ball_order20.mp4 -i v2_ball_order50.mp4 -i v2_ball_order100.mp4 \
-i v1_shading_order2.mp4 -i v1_shading_order3.mp4 -i v1_shading_order4.mp4 -i v1_shading_order6.mp4 -i v1_shading_order10.mp4 -i v1_shading_order20.mp4 -i v1_shading_order50.mp4 -i v1_shading_order100.mp4 \
-i v2_shading_order2.mp4 -i v2_shading_order3.mp4 -i v2_shading_order4.mp4 -i v2_shading_order6.mp4 -i v2_shading_order10.mp4 -i v2_shading_order20.mp4 -i v2_shading_order50.mp4 -i v2_shading_order100.mp4 \
-filter_complex "
  [0:v]scale=256:256,setsar=1[v0];   [1:v]scale=256:256,setsar=1[v1];
  [2:v]scale=256:256,setsar=1[v2];   [3:v]scale=256:256,setsar=1[v3];
  [4:v]scale=256:256,setsar=1[v4];   [5:v]scale=256:256,setsar=1[v5];
  [6:v]scale=256:256,setsar=1[v6];   [7:v]scale=256:256,setsar=1[v7];
  [8:v]scale=256:256,setsar=1[v8];   [9:v]scale=256:256,setsar=1[v9];
  [10:v]scale=256:256,setsar=1[v10]; [11:v]scale=256:256,setsar=1[v11];
  [12:v]scale=256:256,setsar=1[v12]; [13:v]scale=256:256,setsar=1[v13];
  [14:v]scale=256:256,setsar=1[v14]; [15:v]scale=256:256,setsar=1[v15];
  [16:v]scale=256:256,setsar=1[v16]; [17:v]scale=256:256,setsar=1[v17];
  [18:v]scale=256:256,setsar=1[v18]; [19:v]scale=256:256,setsar=1[v19];
  [20:v]scale=256:256,setsar=1[v20]; [21:v]scale=256:256,setsar=1[v21];
  [22:v]scale=256:256,setsar=1[v22]; [23:v]scale=256:256,setsar=1[v23];
  [24:v]scale=256:256,setsar=1[v24]; [25:v]scale=256:256,setsar=1[v25];
  [26:v]scale=256:256,setsar=1[v26]; [27:v]scale=256:256,setsar=1[v27];
  [28:v]scale=256:256,setsar=1[v28]; [29:v]scale=256:256,setsar=1[v29];
  [30:v]scale=256:256,setsar=1[v30]; [31:v]scale=256:256,setsar=1[v31];

  [v0][v1][v2][v3][v4][v5][v6][v7]hstack=8[row1];
  [v8][v9][v10][v11][v12][v13][v14][v15]hstack=8[row2];
  [v16][v17][v18][v19][v20][v21][v22][v23]hstack=8[row3];
  [v24][v25][v26][v27][v28][v29][v30][v31]hstack=8[row4];

  [row1][row2][row3][row4]vstack=4[out]
" -map "[out]" -shortest output.mp4

"""