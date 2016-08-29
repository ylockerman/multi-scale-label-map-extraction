
%HSLIC Loads the HSLIC Superpixes outputted from "multiscale_extraction"
%   This loads the superpixel tree garnered by our method. The low 
%   level details can be accessed through the properties. The class 
%   also includes functions for high level access. 
%


% -------------------------------------------------------------------------------
% MIT License
% 
% Copyright (c) 2016 Yale University
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
% 
% This is based on the source for the paper 
% 
% "Multi-scale label-map extraction for texture synthesis"
% in the ACM Transactions on Graphics (TOG) - Proceedings of ACM SIGGRAPH 2016,
% Volume 35 Issue 4, July 2016 
% by
% Lockerman, Y.D., Sauvage, B., Allegre, 
% R., Dischler, J.M., Dorsey, J. and Rushmeier, H.
% 
% http://graphics.cs.yale.edu/site/publications/multi-scale-label-map-extraction-texture-synthesis
% 
% If you find it useful, please consider giving us credit or citing our paper.   
% -------------------------------------------------------------------------------
% 
% Created on Mon Jul 08 18:38:55 2013
% 
% @author: Yitzchak David Lockerman

classdef HSLIC

    
    properties
        %The SLIC data at the lowest scale
        %This is an array the size of the image, where each element is the
        %(zerobased) index of the atomic scale superpixel.
        atomic_SLIC 
        
        %A cell array with locations for each base superpixels
        %Each element in the array stores the location of the points within
        %each superpixel. Note that, as a matlab array, the elements are
        %accessed by a 1-based-indexing. Thus superpixel 0 is at element 1
        %and so on.
        atomic_SLIC_table 
        
        %The tree of HSLIC superpixels 
        %This is a cell array of top level superpixels in the hierarchy.
        %Each element contains the following properties:
        % * scale: The scale of the SP
        % * list_of_base_superpixels: An array containing the (zerobased)
        % list of atomic superpixels that make up this higher-level 
        % superpixel. 
        % * children: A cell array of child subpixels.  This child list
        % contains the same structure as HSLIC_tree itself.
        HSLIC_tree 
        
        %The list of all scales
        %This is a sorted list of all scales that contain a superpixel
        %transition.
        scale_list 
    end
    
    methods
        function out = HSLIC(filename)
            %Loads the HSLIC superpixels from a saved file. 
            %A labeling file can be created by using the python program
            %"multiscale_extraction." Please see that program for details.
            %This constructer takes the file path of that output file.
            vars = {'image_shape','atomic_SLIC_rle','HSLIC'};
            vars = load(filename,vars{:});

            %load atomic_SLIC fromt the run length encoding used in the file.
            out.atomic_SLIC = zeros(vars.image_shape(1)*vars.image_shape(2),1,'int32');
            upto = 1;
            for ii = 1:size(vars.atomic_SLIC_rle,1)
                out.atomic_SLIC(upto:upto+vars.atomic_SLIC_rle(ii,1)-1) = vars.atomic_SLIC_rle(ii,2);
                upto=upto+vars.atomic_SLIC_rle(ii,1);
            end
            out.atomic_SLIC = reshape(out.atomic_SLIC,vars.image_shape(2),vars.image_shape(1))';

            %Build a lookup table for the diffrent superpixels 
            lookup_table = zeros(3,vars.image_shape(1),vars.image_shape(2));
            lookup_table(1,:,:) = out.atomic_SLIC;
            [lookup_table(3,:,:),lookup_table(2,:,:)] = ...
                      meshgrid(1:vars.image_shape(2),1:vars.image_shape(1));
            
            lookup_table = reshape(lookup_table,3,[]);
            sorted_tables = sortrows(lookup_table')';
            
            [~,ia,~] = unique(sorted_tables(1,:));
            ia = [ia; size(sorted_tables,2)];
            
            out.atomic_SLIC_table = cell(max(max(out.atomic_SLIC+1)),1);
            for ii = 1:(size(ia,1)-1)
                out.atomic_SLIC_table{ii} = sorted_tables(2:end,ia(ii):ia(ii+1)-1);
            end
            
            out.scale_list = get_all_scales(vars.HSLIC);
            out.HSLIC_tree = vars.HSLIC;
        end
        
        function [ indicator ] = superpixels_indicator(obj,scale )
        %SUPERPIXELS_AT_SCALE Returns a table of the superpixels at a level
        %   This will create a 1-based indicator function for all the
        %   superpixels at a given level. The output will be a 2d integer
        %   array with an superpixel index at each location. Note that the
        %   index given is arbitrary.

            indicator = zeros(size(obj.atomic_SLIC));
            [indicator, ~] = make_indicator(indicator,1,obj.HSLIC_tree,obj,scale);
        end
        
    end
    
end

function scale_list = get_all_scales(HSLIC_node_list)
    %Helper function to create the list of all scales recursively. 
    scale_list = [];

    for subnode = HSLIC_node_list
        scale_list = union(scale_list,subnode{1}.scale);
        scale_list = union(scale_list,get_all_scales(subnode{1}.children));
    end

end

function [indicator,next_id] = make_indicator(indicator,next_id,HSLIC_node_list,obj, scale)
    %Helper function to create the indicator function for a scale. 
    
    for subnode = HSLIC_node_list
        if  subnode{1}.scale <= scale || isempty(subnode{1}.children)
            for spid = subnode{1}.list_of_atomic_superpixels 
                indexes = obj.atomic_SLIC_table{spid+1};
                indicator(indexes(1,:),indexes(2,:)) = next_id;
            end
            next_id = next_id + 1;
        else
            [indicator,next_id] = make_indicator(indicator,next_id,subnode{1}.children,obj,scale);
        end
    end
end
