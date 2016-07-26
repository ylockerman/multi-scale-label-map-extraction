# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
MIT License

Copyright (c) 2016 Yale University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is based on the source for the paper 

"Multi-scale label-map extraction for texture synthesis"
in the ACM Transactions on Graphics (TOG) - Proceedings of ACM SIGGRAPH 2016,
Volume 35 Issue 4, July 2016 
by
Lockerman, Y.D., Sauvage, B., Allegre, 
R., Dischler, J.M., Dorsey, J. and Rushmeier, H.

http://graphics.cs.yale.edu/site/publications/multi-scale-label-map-extraction-texture-synthesis

If you find it useful, please consider giving us credit or citing our paper.   
-------------------------------------------------------------------------------

Created on Mon Nov 03 16:04:21 2014

@author: Yitzchak David Lockerman
"""

import numpy as np

from mako.template import Template

import opencl_tools
cl = opencl_tools.cl;
cl_array = opencl_tools.cl_array
cl_reduction = opencl_tools.cl_reduction

if __name__ == '__main__':
    import sys
    sys.path.append('.')

class InKernalMatrix(object):
    def __init__(self,queue,n,eltype='float'):
        assert n in [2,3,4,8,16];            
        assert isinstance(eltype,str)    
        
        if n == 16:
            raise ValueError("n of 16 dose not work for now")
        self.n = n
        ctx = queue.context
        
        self.matrix_name="matrix%dx%d" % (n,n)
        self.vector_type_name = "%s%d" % (eltype,n)
        self.vector_type = getattr(cl_array.vec,self.vector_type_name)
        self.element_type = self.vector_type.fields['x'][0]
        self.element_type_name = eltype
        self.vector_padded = len(self.vector_type.names)
        
        output_defs = []
        
        #Create the structure
        mat_struct = np.dtype([("r%d"%row, self.vector_type) 
                                                for row in xrange(n)])
        mat_struct, mat_struct_c_decl = \
                        cl.tools.match_dtype_to_c_struct(
                                  ctx.devices[0], self.matrix_name, mat_struct)
        output_defs.append(mat_struct_c_decl)                          
        self.mat_struct = cl.tools.get_or_register_dtype(self.matrix_name,
                                                                     mat_struct)
        
        
        #Create code to multiply 
        mult_code = Template("""
        ${vector_type} mat_dot(${matrix_type} mat,${vector_type} v) 
        {
            ${vector_type} out;
            %for row in xrange(rows):
                <%
                        if row <= 9:
                            row_name = row
                        else:
                            row_name = chr(ord('a')+row-10)
                %>

                %if rows <= 4:
                    out.s${row_name} = dot(mat.r${row},v);
                %elif rows == 8:
                    out.s${row_name} = dot(mat.r${row}.lo,v.lo) +
                                       dot(mat.r${row}.hi,v.hi);
                %elif rows == 16:
                    out.s${row_name} = dot(mat.r${row}.lo.lo,v.lo.lo) +
                                       dot(mat.r${row}.lo.hi,v.lo.hi) +
                                       dot(mat.r${row}.hi.lo,v.hi.lo) +
                                       dot(mat.r${row}.hi.hi,v.hi.hi);
                %endif
            %endfor
            
            return out;
        }
        
        ${matrix_type} mat_mul_t(${matrix_type} mat1,${matrix_type} mat2_T) 
        {
            ${matrix_type} out;
            %for row in xrange(rows):
  
                %for col in xrange(rows):
                    <%
                            if col <= 9:
                                col_name = col
                            else:
                                col_name = chr(ord('a')+col-10)
                    %>   
                        out.r${row}.s${col_name} = 
                        %if rows <= 4:
                                 dot(mat1.r${row},mat2_T.r${col});
                        %elif rows == 8:
                                 dot(mat1.r${row}.lo,mat2_T.r${col}.lo) +
                                 dot(mat1.r${row}.hi,mat2_T.r${col}.hi);
                        %elif rows == 16:
                                 dot(mat1.r${row}.lo.lo,mat2_T.r${col}.lo.lo) +
                                 dot(mat1.r${row}.lo.hi,mat2_T.r${col}.lo.hi) +
                                 dot(mat1.r${row}.hi.lo,mat2_T.r${col}.hi.lo) +
                                 dot(mat1.r${row}.hi.hi,mat2_T.r${col}.hi.hi);
                        %endif
                %endfor
            %endfor
            
            return out;
        }     
        
        ${matrix_type} mat_mul(${matrix_type} mat1,${matrix_type} mat2) 
        {
            ${matrix_type} mat2_T;
            %for row in xrange(rows):
                <%
                        if row <= 9:
                            row_name = row
                        else:
                            row_name = chr(ord('a')+row-10)
                %>   
                %for col in xrange(rows):
                    <%
                            if col <= 9:
                                col_name = col
                            else:
                                col_name = chr(ord('a')+col-10)
                    %> 
                    mat2_T.r${row}.s${col_name} = mat2.r${col}.s${row_name};
                %endfor
            %endfor
            
            return mat_mul_t(mat1,mat2_T);
        }
        
        """).render(matrix_type=self.matrix_name,
                    vector_type=self.vector_type_name,
                    rows=n,output_encoding='ascii');
        output_defs.append(mult_code)
        
        if n <= 4:
            inv_code = Template(r"""
                <%!
                    import itertools 

                    def name_for(index):
                        if index <= 9:
                            return "%d" % index
                        else:
                            return chr(ord('a')+index-10) 
                    def sign(perm,perm_0):
                        sign = 1
                        sign_0 = 1
                        for i in xrange(len(perm)):
                            for j in xrange(i):
                                sign*= (perm[i]-perm[j]);
                                sign_0*= (perm_0[i]-perm_0[j]);
                        return sign/sign_0;
                %>
                ${element_type} det(${matrix_type} mat)
                {
                    ${element_type} val = 
                    % for perm in itertools.permutations(xrange(rows)):
                        ${ '+' if sign(perm,range(rows)) > 0 else '-'}
                        % for row in xrange(rows):
                           ${'*' if row > 0 else ''}mat.r${row}.s${name_for(perm[row])}
                        % endfor 
                    % endfor
                    ;
                    return val;
                }
                
                ${matrix_type} inv(${matrix_type} mat,${element_type} det_val )
                {
                    ${matrix_type} out;
                    
                    
                    %for row in xrange(rows):
                        %for col in xrange(rows):
                        out.r${col}.s${name_for(row)} = 1/det_val*(
                            <%
                                rows_of_intrest = filter(lambda x: x != row,xrange(rows));
                                cols_of_intrest = filter(lambda x: x != col,xrange(rows));
                                form_sign = 1 if (row + col) % 2 == 0 else -1; 
                            %>
                            % for perm in itertools.permutations(cols_of_intrest):
                                ${ '+' if form_sign*sign(perm,cols_of_intrest) > 0 else '-'}
                                % for co_factor_row in xrange(rows-1):
                                   ${'*' if co_factor_row > 0 else ''}mat.r${rows_of_intrest[co_factor_row]}.s${name_for(perm[co_factor_row])}
                                % endfor 
                            % endfor
                        );
                        % endfor
                    % endfor
                    
                    return out;
                }     
                
                
                
            """).render(matrix_type=self.matrix_name,
                        vector_type=self.vector_type_name,
                        element_type=self.element_type_name,
                        rows=n,output_encoding='ascii');

            output_defs.append(inv_code)        
        self.preamble_code = str('\n'.join(output_defs))

        
    def np_array_to_vector(self,nparr,out = None):
        assert nparr.size == self.n
        if out is None:
            out = np.empty(1,dtype=self.vector_type)
        else:
            assert out.size == 1 and out.dtype == self.vector_type
        
        out.view(self.element_type)[:n] = nparr
        return out

        
    def np_array_to_matrix(self,nparr,out=None):
        assert nparr.shape == (self.n,self.n)

        if out == None:
            out = np.empty(1,dtype=self.mat_struct)
        else:
            assert out.shape == 1 and out.dtype == self.mat_struct
        mat_view = out.view(self.element_type) 
        mat_view = mat_view.reshape((self.n,self.vector_padded))
        
        mat_view[:self.n,:self.n] = nparr

        return out        
        
        
    def vector_to_np_array(self,vec,out = None):
        assert vec.size == 1
        assert vec.dtype == self.vector_type
        return vec.view(self.element_type)[:n] 

        
    def matrix_to_np_array(self,mat):
        assert mat.size == 1
        assert mat.dtype == self.mat_struct

        mat_view = np.empty((self.n,self.vector_padded),self.element_type)
        mat_view.data[:] = mat.data[:]

        #mat_view = mat.view(self.element_type) 
        #mat_view = mat_view.reshape((self.n,self.vector_padded))
        
        return mat_view[:self.n,:self.n]
   


#        
if __name__ == "__main__":
    queue = cl.CommandQueue(opencl_tools.get_a_context(),
                                properties=opencl_tools.profile_properties)
                                
    test_code = Template("""
    __kernel void test_kernal(${matrix_type} mat_in,
                         ${vector_type} v_in,__global ${vector_type}* v_out) 
    {
        *v_out = mat_dot(mat_in,v_in);
    }
    
    __kernel void test_kernal2(${matrix_type} mat1_in,
                         ${matrix_type} mat2_in,__global ${matrix_type}* m_out) 
    {
        *m_out = mat_mul(mat1_in,mat2_in);
    }   
    
    % if rows <= 4:
        __kernel void test_kernal3(${matrix_type} in, __global ${matrix_type}* m_out) 
        {
            ${element_type} det_val = det(in);
            *m_out = inv(in,det_val);
        }       
    %endif
    """)
                                
    for eltype in ['float']:
        for n in [2,3,4,8]:#16]:
            print "-----------------------",n
            k_matrix = InKernalMatrix(queue,n,eltype)
            
            test_mat = np.random.random((n,n)).astype(dtype=k_matrix.element_type)
            mymat = k_matrix.np_array_to_matrix(test_mat)

            test_mat2 = np.random.random((n,n)).astype(dtype=k_matrix.element_type)
            mymat2 = k_matrix.np_array_to_matrix(test_mat2)

            test_vec = np.random.random((n)).astype(dtype=k_matrix.element_type)
            myvec = k_matrix.np_array_to_vector(test_vec)
            assert np.any(k_matrix.vector_to_np_array(myvec) == test_vec)
            
            program_code = str(test_code.render(matrix_type=k_matrix.matrix_name,
                                            vector_type=k_matrix.vector_type_name,
                                            element_type=eltype,rows = n,
                                            output_encoding='ascii'))
            #print k_matrix.preamble_code+program_code
            prg = cl.Program(queue.context,
                              k_matrix.preamble_code+program_code
                              ).build(options=[]);
            
            ####################Test 1
            test_kernal = prg.test_kernal
            test_kernal.set_scalar_arg_dtypes([
                                k_matrix.mat_struct,#${matrix_type} mat_in,
                                k_matrix.vector_type,#${vector_type} v_in,
                                None,#${vector_type}* v_out
                                ])
                                
                           
                                
            out = cl_array.empty(queue,(1,),k_matrix.vector_type);

            
            test_kernal(queue,(1,),(1,),mymat,myvec,out.data);
            out_cpu = k_matrix.vector_to_np_array(out.get(queue))

            

            assert np.allclose(np.dot(test_mat,test_vec) , out_cpu)
            out.data.release()
            del out
            
            #######################Test 2
            test_kernal2 = prg.test_kernal2
            test_kernal2.set_scalar_arg_dtypes([
                                k_matrix.mat_struct,
                                k_matrix.mat_struct,
                                None,
                                ])     
            out = cl_array.empty(queue,(1,),k_matrix.mat_struct);
            
            test_kernal2(queue,(1,),(1,),mymat,mymat2,out.data);
            out_cpu = k_matrix.matrix_to_np_array(out.get(queue))

            assert np.allclose(np.dot(test_mat,test_mat2) , out_cpu)
            out.data.release()
            del out
            
            #######################Test 3
            if n <= 4:
                test_kernal3 = prg.test_kernal3
                test_kernal3.set_scalar_arg_dtypes([
                                    k_matrix.mat_struct,
                                    None,
                                    ])     
                out = cl_array.empty(queue,(1,),k_matrix.mat_struct);            
                test_kernal3(queue,(1,),(1,),mymat,out.data);
                out_cpu = k_matrix.matrix_to_np_array(out.get(queue))
                

                assert np.allclose(np.linalg.inv(test_mat) , out_cpu)
                out.data.release()
                del out            