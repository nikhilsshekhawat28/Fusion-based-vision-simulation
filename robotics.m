clc;
clear;
close all;

% === ROBOT PARAMETERS ===
L1 = 30;    % Base height
L2 = 50;    % Arm link 1 length
L3 = 40;    % Arm link 2 length
L4 = 20;    % End-effector length

% === ENVIRONMENT SETTINGS ===
hold on
axis equal
grid on
xlabel('X');
ylabel('Y');
zlabel('Z');
view(3)
xlim([-150, 150])
ylim([-150, 150])
zlim([0, 150])

% === CAMERA & LIGHTING ===
camlight('right')
lighting gouraud
light('Position',[-100,100,100],'Style','local')
light('Position',[100,-100,100],'Style','local')

% === TABLE WITH TEXTURE ===
table_length = 150;
table_width = 100;
table_height = 5;
table_color = [0.6, 0.3, 0.1];  % Brown color

% Table surface
fill3([-table_length/2, table_length/2, table_length/2, -table_length/2], ...
      [-table_width/2, -table_width/2, table_width/2, table_width/2], ...
      [0, 0, 0, 0], table_color)

% Table legs
leg_height = 40;
leg_size = 10;
leg_color = [0.4, 0.2, 0.1];

% Four legs
fill3([-table_length/2, -table_length/2 + leg_size, -table_length/2 + leg_size, -table_length/2], ...
      [-table_width/2, -table_width/2, -table_width/2 + leg_size, -table_width/2 + leg_size], ...
      [0, 0, -leg_height, -leg_height], leg_color)

fill3([table_length/2, table_length/2 - leg_size, table_length/2 - leg_size, table_length/2], ...
      [-table_width/2, -table_width/2, -table_width/2 + leg_size, -table_width/2 + leg_size], ...
      [0, 0, -leg_height, -leg_height], leg_color)

fill3([-table_length/2, -table_length/2 + leg_size, -table_length/2 + leg_size, -table_length/2], ...
      [table_width/2, table_width/2, table_width/2 - leg_size, table_width/2 - leg_size], ...
      [0, 0, -leg_height, -leg_height], leg_color)

fill3([table_length/2, table_length/2 - leg_size, table_length/2 - leg_size, table_length/2], ...
      [table_width/2, table_width/2, table_width/2 - leg_size, table_width/2 - leg_size], ...
      [0, 0, -leg_height, -leg_height], leg_color)

% === WALLS & FLOOR FOR ROOM ===
wall_color = [0.8, 0.8, 0.8];  % Light gray
floor_color = [0.5, 0.5, 0.5]; % Darker gray floor

% Floor
fill3([-200, 200, 200, -200], [-200, -200, 200, 200], [-leg_height, -leg_height, -leg_height, -leg_height], floor_color)

% Back wall
fill3([-200, 200, 200, -200], [-200, -200, -200, -200], [-leg_height, -leg_height, 150, 150], wall_color)

% Side wall
fill3([-200, -200, -200, -200], [-200, 200, 200, -200], [-leg_height, -leg_height, 150, 150], wall_color)

% === UPRIGHT DUSTBIN ===
bin_radius = 15;
bin_height = 40;

% Dustbin cylinder (upright)
[x, y, z] = cylinder(bin_radius, 30);
z = z * bin_height;
x = x + 80;    % X position
y = y + 50;    % Y position

surf(x, y, z, 'FaceColor', 'b', 'EdgeColor', 'none')

% === TCP SERVER FOR YOLO COORDINATES ===
port = 5005;
server = tcpserver('127.0.0.1', port);
disp('Waiting for YOLO coordinates...');

% === OBJECT HANDLE FOR MOVEMENT ===
X_object = 0;
Y_object = 0;
Z_object = 5;

object_handle = plot3(X_object, Y_object, Z_object, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% === ROBOT ARM LINKS ===
L(1) = Link([0, 0, L1, pi/2]);
L(2) = Link([0, 0, L2, 0]);
L(3) = Link([0, 0, L3, 0]);
L(4) = Link([0, 0, L4, 0]);

Robot = SerialLink(L, 'name', 'RealisticArm');

% === TCP Listener Loop ===
while true
    % Wait for YOLO data
    if server.NumBytesAvailable >= 8
        data = read(server, 8, 'uint8');
        
        % Decode YOLO coordinates
        x_norm = typecast(uint8(data(1:4)), 'single');
        y_norm = typecast(uint8(data(5:8)), 'single');

        % === MAP YOLO COORDINATES TO ROBOT SPACE ===
        image_width = 640;    % YOLO frame width
        image_height = 480;   % YOLO frame height
        workspace_x = 150;    % Robot X range
        workspace_y = 150;    % Robot Y range

        X_robot = (x_norm * workspace_x) - (workspace_x / 2) + 30;
        Y_robot = (y_norm * workspace_y) - (workspace_y / 2) + 30;

        % === UPDATE OBJECT POSITION ===
        set(object_handle, 'XData', X_robot, 'YData', Y_robot, 'ZData', 5);
        
        % === MOVE ARM TO OBJECT ===
        T1 = transl(X_robot, Y_robot, Z_object);
        q1 = Robot.ikine(T1, 'mask', [1 1 1 0 0 0]);
        
        if any(isnan(q1))
            warning('Failed to converge to the object.');
        else
            Robot.plot(q1);
            pause(1);
        end

        % === SIMULATE PICKING THE OBJECT ===
        Z_pick = Z_object + 10;
        T_pick = transl(X_robot, Y_robot, Z_pick);
        q_pick = Robot.ikine(T_pick, 'mask', [1 1 1 0 0 0]);
        
        if ~any(isnan(q_pick))
            Robot.plot(q_pick);
            
            % Simulate object being lifted
            set(object_handle, 'XData', X_robot, 'YData', Y_robot, 'ZData', Z_pick);
            pause(1);
        end

        % === MOVE TO UPRIGHT DUSTBIN ===
        X_bin = 80;
        Y_bin = 50;
        Z_bin = 20;

        T_bin = transl(X_bin, Y_bin, Z_bin);
        q_bin = Robot.ikine(T_bin, 'mask', [1 1 1 0 0 0]);
        
        if ~any(isnan(q_bin))
            Robot.plot(q_bin);
            pause(1);
            
            set(object_handle, 'XData', X_bin, 'YData', Y_bin, 'ZData', Z_bin);
            pause(1);
        end

        Robot.plot([0, 0, 0, 0]);
    end
end
