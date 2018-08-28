function wcvt_2d_sampling ( g_num, it_num, s_num, wts)

  timestamp ( );
  fprintf ( 1, '\n' );
  fprintf ( 1, 'CVT_2D_SAMPLING\n' );
  fprintf ( 1, '  MATLAB version\n' );
  fprintf ( 1, '  Use sampling to approximate Lloyd''s algorithm\n' );
  fprintf ( 1, '  in the 2D unit square.\n' );

  if ( nargin < 1 )
    g_num = input ( '  Enter number of generators: ' );
  elseif ( ischar ( g_num ) )
    g_num = str2num ( g_num );
  end

  if ( nargin < 2 ) 
    it_num = input ( '  Enter number of iterations: ' );
  elseif ( ischar ( it_num ) )
    it_num = str2num ( it_num );
  end

  if ( nargin < 3 ) 
    s_num = input ( '  Enter number of sample points: ' );
  elseif ( ischar ( s_num ) )
    s_num = str2num ( s_num );
  end
  
  if ( nargin < 4)
      wts = [0.1;zeros(g_num-1,1)];
  end
  wts = wts - max(wts);
  
  fprintf ( 1, '\n' );
  fprintf ( 1, '  Number of generators is %d\n', g_num );
  fprintf ( 1, '  Number of iterations is %d\n', it_num );
  fprintf ( 1, '  Number of samples is %d\n', s_num );
%
%  Initialize the generators.
%
  g = rand ( 2, g_num );
%
%  Carry out the iteration.
%
  step = 1 : it_num;
  e = nan ( it_num, 1 );
  gm = nan ( it_num, 1 );

  for it = 1 : it_num
%
%  Compute the Delaunay triangle information T for the current nodes.
%
    t = wdelaunay(g(1,:),g(2,:),wts);
    [PC,~] = powercenters(t,g(1,:),g(2,:),wts);
%
%  Display the Voronoi cells.
%
    subplot ( 2, 2, 1 );    
    plot(PC(:,1), PC(:,2), 'r.');
    hold on;
    powerdiagram(g(1,:),g(2,:),t,'r',wts);
    plot(g(1,:),g(2,:), 'b.');
    title_string = sprintf ( 'Weighted Voronoi, step %d', it );
    title ( title_string );
    axis equal
    axis ([  0.0, 1.0, 0.0, 1.0 ])
    axis square
    drawnow
    hold off;
%
%  Display the Delaunay triangulation.
%
    subplot ( 2, 2, 2 );
    trimesh (t,g(1,:),g(2,:),zeros(g_num,1), 'EdgeColor', 'r' )
    hold on
    scatter ( g(1,:), g(2,:), 'k.' )
    title_string = sprintf ( 'Weighted Delaunay, step %d', it );
    title ( title_string );
    axis ( [  0.0, 1.0, 0.0, 1.0 ] )
    axis square
    view ( 2 )
    hold off
%
%  Generate sample points.
%  New option for fixed grid sampling.
%
  s2 = floor ( sqrt ( s_num ) );
  s2 = max ( s2, 2 );
  s_num = s2 * s2;
  [ sx, sy ] = meshgrid ( linspace ( 0.0, 1.0, s2 ) );
  sx = reshape ( sx, 1, s2 * s2 );
  sy = reshape ( sy, 1, s2 * s2 );
  s = [ sx; sy ];
%
%  For each sample point, find K, the index of the nearest generator.
%  We do this efficiently by using the Delaunay information with
%  Matlab's DSEARCH command, rather than a brute force nearest neighbor
%  computation.
%  
    k = powercellidx (g(1,:),g(2,:),s(1,:),s(2,:),wts);

    m = accumarray ( k, ones(s_num,1) );
    g_new = g;
    
    sumx = accumarray ( k, s(1,:) );
    sumy = accumarray ( k, s(2,:) );
    g_new(1,m~=0) = sumx(m~=0) ./ m(m~=0);
    g_new(2,m~=0) = sumy(m~=0) ./ m(m~=0);
    
%
%  Compute the energy.
%
    e(it,1) = (sum ( ( s(1,:) - g(1,k(:)) ).^2 ...
                  + ( s(2,:) - g(2,k(:)) ).^2 -wts(k(:))')) / s_num;
              
%
%  Display the energy.
%
    subplot ( 2, 2, 3 )
    plot ( step, log ( e ), 'm-*' )
    title ( 'Log (Energy)' )
    xlabel ( 'Step' )
    ylabel ( 'Energy' )
    grid
%
%  Compute the generator motion.
%
    gm(it,1) = sum ( ( g_new(1,:) - g(1,:) ).^2 ...
                   + ( g_new(2,:) - g(2,:) ).^2 ) / g_num;
%
%  Display the generator motion.
%
    subplot ( 2, 2, 4 )
    plot ( step, log ( gm ), 'm-*' )
    title ( 'Log (Average generator motion)' )
    xlabel ( 'Step' )
    ylabel ( 'Energy' )
    grid
%
%  Continue?

%
%     s = input ( 'RETURN, or Q to quit: ', 's' );
% 
%     if ( s == 'q' | s == 'Q' )
%       break
%     end
pause(1);

%
%  Update the generators.
%
    g = (g_new+g)/2;

  end
%
%  Terminate.
%
  fprintf ( 1, '\n' );
  fprintf ( 1, 'CVT_2D_SAMPLING\n' );
  fprintf ( 1, '  Normal end of execution.\n' );
  fprintf ( 1, '\n' );
  timestamp ( );

  return
end
function timestamp ( )

%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    14 February 2003
%
%  Author:
%
%    John Burkardt
%
  t = now;
  c = datevec ( t );
  s = datestr ( c, 0 );
  fprintf ( 1, '%s\n', s );

  return
end