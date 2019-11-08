function [x,y] = poincare_index(orientation)
x = [];
y = [];



orientation = rad2deg(orientation);
for i = 2:size(orientation,1)-1
    for j = 2:size(orientation,2)-1
        cells = [[-1, -1]; [-1, 0]; [-1, 1]; [0, 1]; [1, 1]; [1, 0]; [1, -1]; [0, -1]; [-1, -1]];
        angles = zeros(9);
        for k = 1:length(cells)
            i1 = i+cells(k,1);
            j1 = j+cells(k,2);
            angles(k) = orientation(i1,j1);
        end
        index = 0;
        for k = 1:length(cells)
            if abs(angles(k)-angles(k+1))>90
                angles(k+1) = angles(k+1) + 180;
            end
            index = index+ angles(k)-angles(k+1);
        end
        if 179.5<=abs(index)&&180.5>=abs(index)
            x = [x i];
            y = [y j];
        end
    end
end