(() => {
  const sequenceEvents = [
    'run_start', 'device_ready', 'pipeline_ready', 'node_ready', 'node_start',
    'inputs_ready', 'buffer_allocated', 'input_buffer_created', 'input_buffer_created',
    'output_buffer_created', 'command_buffer_created', 'encoder_configured',
    'dispatch_configured', 'kernel_submitted', 'kernel_completed', 'copy_back_started',
    'copy_back_completed', 'node_complete', 'node_ready', 'node_start', 'inputs_ready',
    'buffer_allocated', 'input_buffer_created', 'input_buffer_created',
    'output_buffer_created', 'command_buffer_created', 'encoder_configured',
    'dispatch_configured', 'kernel_submitted', 'kernel_completed', 'copy_back_started',
    'copy_back_completed', 'node_complete', 'run_complete'
  ];
  const codeLines = [
    'Graph graph;', 'graph.set_input', 'graph.set_input', 'graph.set_input',
    'graph.add_node(MatMul', 'graph.add_node(Add', 'MemoryPool pool;',
    'MetalBackend backend;', 'Executor executor', 'executor.run',
    'newBufferWithBytes', 'newBufferWithBytes', 'dispatchThreads',
    'hidden[row,col]', 'dispatchThreads', 'out[index]', 'waitUntilCompleted',
    'memcpy'
  ];

  function lineFor(event, detail) {
    if (event === 'run_start') return [0, 1, 2, 3];
    if (event === 'node_ready') return detail.includes('matmul') ? [4, 8, 9] : [5, 8, 9];
    if (event === 'node_start' || event === 'inputs_ready') return [8, 9];
    if (event === 'buffer_allocated') return [6, 9];
    if (event === 'input_buffer_created') return [10, 11];
    if (event === 'output_buffer_created') return [6, 10, 11];
    if (event === 'command_buffer_created' || event === 'encoder_configured') return [9, 10, 11];
    if (event === 'dispatch_configured' || event === 'kernel_submitted') return detail.includes('matmul') ? [12, 13] : [14, 15];
    if (event === 'kernel_completed') return detail.includes('matmul') ? [13] : [15];
    if (event === 'copy_back_started' || event === 'copy_back_completed') return [16, 17];
    if (event === 'node_complete') return detail === 'hidden' ? [4, 9, 17] : [5, 9, 17];
    if (event === 'run_complete') return [17];
    return [0];
  }

  function current() {
    const counter = document.getElementById('counter') || document.getElementById('eventNo');
    const number = Number((counter?.textContent || '1').split('/')[0].trim()) || 1;
    const event = sequenceEvents[Math.max(0, Math.min(sequenceEvents.length - 1, number - 1))];
    const detail = document.getElementById('kernel')?.textContent || '';
    return { number, event, detail, lines: lineFor(event, detail) };
  }

  function sync() {
    const state = current();
    const styleId = 'code-event-sync-style';
    if (!document.getElementById(styleId)) {
      const style = document.createElement('style');
      style.id = styleId;
      style.textContent = '.code-lines span.related,.code span.related{background:#1c3657;border-left-color:#72e5df!important}.event.related{outline:1px solid #72e5df55;background:#18304c}.event.current{outline:2px solid #ffc46d;background:#5d482d}';
      document.head.appendChild(style);
    }
    document.querySelectorAll('#code span,#code-lines span,#codeLarge span').forEach((line, index) => {
      line.classList.toggle('related', state.lines.includes(index));
      line.classList.toggle('active', state.lines.includes(index));
    });
    document.querySelectorAll('#trace .event').forEach((item, index) => {
      const eventName = item.querySelector('b')?.textContent || '';
      const detail = item.querySelector('span')?.textContent || '';
      const related = lineFor(eventName, detail).some(line => state.lines.includes(line));
      item.classList.toggle('related', related);
      item.classList.toggle('current', index === state.number - 1);
    });
  }

  setInterval(sync, 100);
})();
